"""重置纸面账户：清空持仓/成交/净值，资金恢复初始资金。

供开市前清理试验数据使用（不影响日线/分钟行情数据）。

用法：
    python scripts/09_reset_accounts.py          # 重置自动纸面盘 + 模拟炒股
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.config import load_config            # noqa: E402
from quant.trading.paper import PaperBroker     # noqa: E402


def main():
    cfg = load_config()
    targets = [
        ("自动纸面盘", cfg.resolve("paper/paper_account.db"),
         cfg["backtest"]["initial_capital"]),
        ("模拟炒股", cfg.resolve(cfg["manual"]["db_path"]),
         cfg["manual"]["initial_capital"]),
    ]
    for name, db_path, init in targets:
        broker = PaperBroker(db_path, initial_capital=init,
                             commission=cfg["backtest"]["commission"],
                             slippage=cfg["backtest"]["slippage"])
        broker.reset()
        s = broker.account_summary()
        print(f"[{name}] 已重置 -> 资金 {s['cash']:.2f} | 持仓 "
              f"{len(broker.query_positions())} | 成交 {len(broker.trade_history(limit=1000))} 笔")
    print("\n完成。日线/分钟行情数据已保留。")


if __name__ == "__main__":
    main()
