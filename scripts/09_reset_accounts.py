"""重置账户：清空持仓/成交/净值，资金恢复初始资金。

供开市前清理试验数据使用（不影响日线/分钟行情数据）。

用法：
    python scripts/09_reset_accounts.py                 # 重置自动纸面盘 + 模拟炒股 + 实盘
    python scripts/09_reset_accounts.py --only real     # 只重置实盘账户
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.config import load_config                  # noqa: E402
from quant.trading.paper import PaperBroker           # noqa: E402
from quant.trading.real_account import RealBroker     # noqa: E402


def main():
    cfg = load_config()
    q = cfg["backtest"]
    real = cfg.get("real", {}) or {}
    targets = [
        ("auto", "自动纸面盘", cfg.resolve("paper/paper_account.db"),
         cfg["backtest"]["initial_capital"], dict(
             commission=q["commission"], slippage=q["slippage"],
             stamp_tax=q.get("stamp_tax", 0.0005))),
        ("manual", "模拟炒股", cfg.resolve(cfg["manual"]["db_path"]),
         cfg["manual"]["initial_capital"], dict(
             commission=q["commission"], slippage=q["slippage"],
             stamp_tax=q.get("stamp_tax", 0.0005),
             lot_size=int(cfg["manual"].get("lot_size", 100)))),
        ("real", "实盘(¥3000)", cfg.resolve(real.get("db_path", "paper/real_account.db")),
         float(real.get("initial_capital", 3000.0)), dict(
             commission=float(real.get("commission", q["commission"])),
             slippage=float(real.get("slippage", 0.0)),
             stamp_tax=float(real.get("stamp_tax", q.get("stamp_tax", 0.0005))),
             lot_size=int(real.get("lot_size", 100)),
             min_commission=float(real.get("min_commission", 0.0)),
             transfer_fee=float(real.get("transfer_fee", 0.0)))),
    ]

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="只重置指定账户：auto / manual / real（默认全部）")
    args = ap.parse_args()

    for key, name, db_path, init, kw in targets:
        if args.only and args.only != key:
            continue
        cls = RealBroker if key == "real" else PaperBroker
        broker = cls(db_path, initial_capital=init, **kw)
        broker.reset()
        s = broker.account_summary()
        print(f"[{name}] 已重置 -> 资金 {s['cash']:.2f} | 持仓 "
              f"{len(broker.query_positions())} | 成交 {len(broker.trade_history(limit=1000))} 笔")
    print("\n完成。日线/分钟行情数据已保留。"
          "（实盘账户的订单留痕 real_order_log 属审计数据，重置不清除）")


if __name__ == "__main__":
    main()
