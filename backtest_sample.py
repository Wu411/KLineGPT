import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# 假设 config 依然在，若不需要可注释掉
# from config import CONFIG 

class Backtest:
    def __init__(self, top_k=200, drop_n=10, min_hold_days=5, cost_rate=0.0015, 
                 start_date=None, end_date=None):
        """
        论文风格回测（对齐 Kronos 投资模拟设定的核心约束）：
        - t 日信号排序
        - 用 t -> t+1 close-to-close 收益计当日收益
        - long-only top-k 等权
        - drop-n：每天最多换 n 只股票
        - 最短持有期 min_hold_days
        - 单边交易成本 cost_rate
        
        新增参数:
        - start_date: 回测开始日期 (str, e.g., '2022-01-01')，None表示从头开始
        - end_date: 回测结束日期 (str, e.g., '2022-12-31')，None表示直到最后
        """
        self.top_k = int(top_k)
        self.drop_n = int(drop_n)
        self.min_hold_days = int(min_hold_days)
        self.cost_rate = float(cost_rate)

        # 时间段控制
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None

        self.scores = None
        self.close_prices = None
        self.stock_returns = None
        self.benchmark = None
        self.portfolio_returns = None

        # 运行中状态
        self.holdings = set()
        self.hold_days = {}  # code -> 持有天数（从 1 开始计）


    # ======================================================
    # 1. 数据加载
    # ======================================================
    def load_data(self, score_folder, kline_file, index_file):
        print("正在加载数据...")

        # -------- 1) 信号 --------
        score_list = []
        # 增加容错，检查文件夹是否存在
        if not os.path.exists(score_folder):
            raise ValueError(f"信号文件夹不存在: {score_folder}")
            
        files = sorted([f for f in os.listdir(score_folder) if f.endswith(".csv")])

        for filename in files:
            date_str = os.path.splitext(filename)[0]
            try:
                dt = pd.to_datetime(date_str)
            except ValueError:
                continue
            
            # 简单的预过滤：如果文件名日期明显在范围外，可以跳过读取以加速（可选）
            # 这里为了保险起见，建议还是先读再对齐，或者只做粗略筛选
            if self.start_date and dt < self.start_date:
                continue
            if self.end_date and dt > self.end_date:
                continue

            df = pd.read_csv(os.path.join(score_folder, filename))
            if df.shape[1] < 2:
                continue

            code_col = df.columns[0]
            score_col = df.columns[1]

            s = df.set_index(code_col)[score_col]
            s.name = dt
            score_list.append(s)

        if len(score_list) == 0:
            raise ValueError("未找到有效信号文件（请检查路径或日期范围）")

        self.scores = pd.concat(score_list, axis=1).T.sort_index()

        # -------- 2) K 线（Close 价）--------
        df_all = pd.read_csv(kline_file, parse_dates=['dt'])
        df_all.columns = [c.lower() for c in df_all.columns]

        if 'kdcode' in df_all.columns:
            df_all.rename(columns={'kdcode': 'code'}, inplace=True)

        if 'close' not in df_all.columns:
            raise ValueError("K线文件必须包含 close 列")

        self.close_prices = df_all.pivot(
            index='dt', columns='code', values='close'
        ).sort_index()

        # 用全量 close 先算好 t->t+1 收益，避免最后一天对齐问题
        # 注意：这里必须用全量算，否则切片后的最后一天无法计算 t+1 收益
        self.stock_returns = self.close_prices.pct_change().shift(-1)

        # -------- 3) 基准（对齐 t->t+1 收益）--------
        bench_df = pd.read_csv(index_file, index_col=0, parse_dates=True)
        cols = [c.lower() for c in bench_df.columns]

        if 'close' in cols:
            bench_close = bench_df.iloc[:, cols.index('close')]
        else:
            bench_close = bench_df.iloc[:, 0]

        self.benchmark = bench_close.pct_change().shift(-1).fillna(0)

        # -------- 4) 对齐与时间过滤 --------
        common_dates = (
            self.scores.index
            .intersection(self.close_prices.index)
            .intersection(self.benchmark.index)
        )

        # [修改点]：在此处进行时间段过滤
        if self.start_date:
            common_dates = common_dates[common_dates >= self.start_date]
        if self.end_date:
            common_dates = common_dates[common_dates <= self.end_date]

        if len(common_dates) == 0:
            raise ValueError(f"在指定时间段 {self.start_date} - {self.end_date} 内，信号/行情/基准无交集")

        self.scores = self.scores.loc[common_dates]
        # 注意：close_prices 切片后，仅用于后续可能的逻辑，收益率 stock_returns 已经算好了
        self.close_prices = self.close_prices.loc[common_dates] 
        self.benchmark = self.benchmark.loc[common_dates]
        self.stock_returns = self.stock_returns.loc[common_dates]

        print(f"数据加载完成，回测区间: {common_dates[0].date()} 至 {common_dates[-1].date()}，有效交易日 {len(common_dates)} 天")


    # ======================================================
    # 2. 回测核心逻辑（对齐论文的 top-k / drop-n / min-hold / cost）
    # ======================================================
    def _equal_weight(self, holdings):
        """返回等权权重 dict：code -> weight"""
        if len(holdings) == 0:
            return {}
        w = 1.0 / len(holdings)
        return {c: w for c in holdings}

    def _turnover(self, w_old, w_new):
        """计算单边成本对应的成交额比例：sum(|delta_w|)"""
        keys = set(w_old.keys()) | set(w_new.keys())
        return float(sum(abs(w_new.get(k, 0.0) - w_old.get(k, 0.0)) for k in keys))

    def run_backtest(self):
        """
        每日流程（t 日收盘后调仓，收益按 t->t+1 close-to-close 计）
        """
        dates = list(self.scores.index)

        daily_ret = []
        daily_dates = []

        self.holdings = set()
        self.hold_days = {}

        w_prev = {}  # 上一日调仓后的等权权重
        
        # 记录：第一天建仓会产生从 0 到 1 的换手，计算一次满仓买入成本
        
        for dt in dates:
            score_row = self.scores.loc[dt]
            score_row = score_row.dropna()

            # 只在可交易股票集合里排序（存在收益数据）
            ret_row = self.stock_returns.loc[dt]
            tradable = ret_row.dropna().index
            score_row = score_row.loc[score_row.index.intersection(tradable)]

            if score_row.empty:
                # 无信号或无可交易收益，保持仓位不变，仅计算收益
                w_new = self._equal_weight(self.holdings)
                gross = 0.0
                if len(self.holdings) > 0:
                    r = ret_row.reindex(list(self.holdings)).fillna(0.0)
                    gross = float(r.mean())
                cost = self.cost_rate * self._turnover(w_prev, w_new)
                net = gross - cost
                daily_ret.append(net)
                daily_dates.append(dt)
                w_prev = w_new
                # 持有天数推进
                for c in list(self.holdings):
                    self.hold_days[c] = self.hold_days.get(c, 0) + 1
                continue

            desired = list(score_row.sort_values(ascending=False).head(self.top_k).index)
            desired_set = set(desired)

            # 1) 先更新“可卖出”集合：不在 desired 且持有天数满足 min_hold_days
            to_sell_candidates = []
            for c in self.holdings:
                hd = self.hold_days.get(c, 0)
                if (c not in desired_set) and (hd >= self.min_hold_days):
                    to_sell_candidates.append(c)

            # 卖出优先级：越不在 desired 越应卖出。按 score 从低到高卖出
            def get_score(code):
                return float(score_row.get(code, -1e18))

            to_sell_candidates.sort(key=get_score) 
            to_sell = to_sell_candidates[: self.drop_n]

            # 执行卖出
            for c in to_sell:
                self.holdings.remove(c)
                self.hold_days.pop(c, None)

            # 2) 买入：从 desired 中挑选当前未持有的，最多 drop_n，并且不超过 top_k 容量
            slots = max(0, self.top_k - len(self.holdings))
            buy_limit = min(self.drop_n, slots)

            to_buy = []
            for c in desired:
                if c not in self.holdings:
                    to_buy.append(c)
                if len(to_buy) >= buy_limit:
                    break

            for c in to_buy:
                self.holdings.add(c)
                self.hold_days[c] = 1  # 新买入从 1 天开始计

            # 3) 等权权重（调仓后）
            w_new = self._equal_weight(self.holdings)

            # 4) 计算 gross 收益（t->t+1）
            gross = 0.0
            if len(self.holdings) > 0:
                r = ret_row.reindex(list(self.holdings)).fillna(0.0)
                gross = float(r.mean())

            # 5) 交易成本：单边 cost_rate * sum(|delta_w|)
            cost = self.cost_rate * self._turnover(w_prev, w_new)
            net = gross - cost

            daily_ret.append(net)
            daily_dates.append(dt)

            # 6) 推进持有天数（对留存仓位 +1；新仓位已设为 1）
            for c in list(self.holdings):
                if c not in to_buy:  # 留存仓位
                    self.hold_days[c] = self.hold_days.get(c, 0) + 1

            w_prev = w_new

        self.portfolio_returns = pd.Series(daily_ret, index=pd.to_datetime(daily_dates)).dropna()
        self.benchmark = self.benchmark.loc[self.portfolio_returns.index]

        print("回测计算完成")
        return self.portfolio_returns


    # ======================================================
    # 3. 指标计算
    # ======================================================
    def calculate_metrics(self):
        rp = self.portfolio_returns
        rb = self.benchmark
        
        if rp.empty:
            return {}, pd.Series()

        cumulative_returns = (1 + rp).cumprod()
        cumulative_bench = (1 + rb).cumprod()

        cr = cumulative_returns.iloc[-1] - 1
        total_ret = cr
        total_bench_ret = cumulative_bench.iloc[-1] - 1

        n_years = len(rp) / 252
        arr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
        bench_arr = (1 + total_bench_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
        aer = arr - bench_arr

        avol = rp.std() * np.sqrt(252)
        roll_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - roll_max) / roll_max
        mdd = drawdown.min()

        asr = (rp.mean() * 252) / (rp.std() * np.sqrt(252)) if rp.std() > 0 else 0
        excess_ret = rp - rb
        ir = (excess_ret.mean() * 252) / (excess_ret.std() * np.sqrt(252)) if excess_ret.std() > 0 else 0

        metrics = {
            "ARR": arr,
            "AER": aer,
            "AVol": avol,
            "MDD": mdd,
            "ASR": asr,
            "IR": ir,
            "CR": cr,
        }
        return metrics, cumulative_returns


    # ======================================================
    # 4. 绘图
    # ======================================================
    def plot_results(self, save_path=None, title="Cumulative Return", ylabel="Return (%)"):
        metrics, cum_ret = self.calculate_metrics()
        if cum_ret.empty:
            print("无可绘图数据")
            return
            
        cum_bench = (1 + self.benchmark).cumprod()

        plt.figure(figsize=(12, 6))
        plt.plot(cum_ret.index, (cum_ret - 1) * 100, label="Strategy", linewidth=2.5, color='red')
        plt.plot(cum_bench.index, (cum_bench - 1) * 100, label="Benchmark", linestyle="--")

        plt.legend()
        plt.grid(True)
        plt.title(f"{title} ({self.start_date.date() if self.start_date else 'Start'} - {self.end_date.date() if self.end_date else 'End'})")
        plt.ylabel(ylabel)

        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":

    BASE_SCORE_FOLDER = "./gpt_csi300/val_scores"
    KLINE_FILE = "../data/output_kline/csi300.csv"
    INDEX_FILE = "../data/output_index_data/csi300_index.csv"
    SAVE_DIR = "./gpt_csi300/backtest/"
    os.makedirs(SAVE_DIR, exist_ok=True)

    epoch_dirs = sorted([
        d for d in os.listdir(BASE_SCORE_FOLDER)
        if d.startswith("epoch_") and os.path.isdir(os.path.join(BASE_SCORE_FOLDER, d))
    ])

    best_cr = float('-inf')
    best_epoch = None
    best_metrics = None

    # === [设置] 指定回测时间段 ===
    START_DATE = "2025-01-01"  # 或 None
    END_DATE = "2025-12-31"    # 或 None

    for epoch_dir in epoch_dirs:
        score_folder = os.path.join(BASE_SCORE_FOLDER, epoch_dir)
        print(f"\n=== 回测 {score_folder} ===")
        print(f"时间段: {START_DATE} ~ {END_DATE}")

        # 在这里传入时间参数
        bt = Backtest(
            top_k=50, 
            drop_n=5, 
            min_hold_days=5, 
            cost_rate=0,
            start_date=START_DATE,
            end_date=END_DATE
        )

        # bt = Backtest(
        #     top_k=200, 
        #     drop_n=10, 
        #     min_hold_days=5, 
        #     cost_rate=0,
        #     start_date=START_DATE,
        #     end_date=END_DATE
        # )

        try:
            bt.load_data(score_folder, KLINE_FILE, INDEX_FILE)
            bt.run_backtest()
            metrics, _ = bt.calculate_metrics()
            print(f"{epoch_dir} metrics:", metrics)

            save_path = os.path.join(SAVE_DIR, f"{epoch_dir}_backtest_plot.png")
            bt.plot_results(save_path=save_path)

            if metrics["CR"] > best_cr:
                best_cr = metrics["CR"]
                best_epoch = epoch_dir
                best_metrics = metrics

        except Exception as e:
            print(f"Error in {epoch_dir}: {e}")
            import traceback
            traceback.print_exc()

    if best_epoch is not None:
        print(f"\n收益最好的epoch: {best_epoch}")
        print("指标:", best_metrics)
    else:
        print("\n没有有效的回测结果。")