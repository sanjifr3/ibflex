from dataclasses import dataclass
import pandas as pd
from typing import Optional, List, Tuple
from ibflex import client, parser, Types
import datetime as dt


def get_data(
    token: int, query_id: int, parse: bool = True, account_id_mapper: Optional[dict] = None
) -> list:
    """Get the data from IBKR Flex"""
    resp = client.download(token, query_id)
    data = parser.parse(resp)
    return parse_data(data, account_id_mapper) if parse else data


def parse_data(data, account_id_mapper: Optional[dict] = None):
    """Parse the data from IBKR Flex"""
    return [IBKR(stmt, account_id_mapper) for stmt in data.FlexStatements]


def clean_symbol(symbol: str, currency: str) -> str:
    """Clean the symbol to match Yahoo Finance"""

    if not symbol:
        return None

    symbol = symbol.replace(".", "-").replace(" ", "-")

    if symbol in ["AUSA", "USD-CAD", "CAD-USD"]:
        return symbol
    elif symbol in ["ATT", "TNY", "PBIC", "AFI"]:
        return symbol + ".CN"
    elif symbol in ["FLT", "DOC"]:
        return symbol + ".V"
    elif symbol in ["CHV-NEW", "CHV"]:
        return "CHV.CN"
    elif symbol in ["CHC"]:
        return "CHC.CN"
    elif symbol == "CGX---220916C00011000":
        return "CGX 16SEP22 11-0 C.TO"
    elif symbol == "ATD":
        return "ATD-B.TO"
    elif symbol == "FB":
        return "META"
    elif currency == "USD":
        return symbol
    return symbol + ".TO"


@dataclass
class Position:
    """Class to hold the position data."""

    ticker: str
    name: str
    currency: str
    category: str
    side: str
    exchange: str
    qty: float
    book_price: float
    cost_basis: float
    market_price: float
    market_value: float
    unreal_gain: float
    pct_of_nav: float
    put_call: Optional[str]
    strike: Optional[float]
    expiry_date: Optional[dt.datetime]
    vesting_date: Optional[dt.datetime]
    last_updated: dt.datetime


class ParseError(ValueError):
    """Parsing error"""

    pass


class IBKR:
    """Class to parse and store IBKR Flex data."""

    TOL = 0.1

    TRANS_COLS = [
        "datetime",
        "site",
        "action",
        "type",
        "ticker",
        "qty",
        "total",
        "fees",
        "dividend",
        "currency",
    ]

    EXPECTED_CASH_TRANS_TYPES = set(
        [
            "WHTAX",
            "BROKERINTPAID",
            "BROKERINTRCVD",
            "FEES",
            "DIVIDEND",
            "PAYMENTINLIEU",
            "DEPOSITWITHDRAW",
        ]
    )

    def __init__(self, data: Types.FlexStatement, account_id_mapper: Optional[dict] = None) -> None:

        # Parse + preprocess
        if account_id_mapper is None:
            account_id_mapper = {}

        self.account_id = account_id_mapper.get(data.accountId, data.accountId)
        self.start_date = data.fromDate
        self.end_date = data.toDate
        self.gen_date = data.whenGenerated
        self.period = data.period

        self._get_account_info(data.AccountInformation)
        self.positions = self._parse_positions(data.OpenPositions)
        self.summ = self._parse_cash_report(data.CashReport)
        self.trades_df, self.fx_df = self._parse_trades(data.Trades)
        self.corp_df = self._parse_corp_actions(data.CorporateActions)
        self.tsfr_df = self._parse_transfers(data.Transfers)  # TODO
        self.cash_trans_df = self._parse_cash_transactions(data.CashTransactions)  # TODO

        # Create transactions dataframe
        self.trans_df = self._create_trans_df()

        # Validate transaction dataframe (should match cash summary)
        # If no data present, skip validation
        if len(self.trans_df) > 0:
            self._validate_data()

    def __repr__(self) -> str:
        repr = f"IBKR | {self.account_id} | {self.start_date.strftime('%Y-%m-%d')} - {self.end_date.strftime('%Y-%m-%d')}"
        if "BASE_SUMMARY" in self.summ:
            repr += f" | {self.summ['BASE_SUMMARY']['end']:.2f} "
        return repr

    def _create_trans_df(self):
        """Create the transactions dataframe."""
        trans_df = pd.concat(
            [self.trades_df, self.fx_df, self.corp_df, self.cash_trans_df, self.tsfr_df],
            axis=0,
            ignore_index=True,
        )

        if len(trans_df) == 0:
            return pd.DataFrame(columns=self.TRANS_COLS)

        trans_df.loc[:, "datetime"] = trans_df.loc[:, "date"].apply(lambda x: x.to_pydatetime())
        trans_df.loc[:, "site"] = self.account_id
        trans_df.loc[trans_df["ticker"] == "MX", "src_ticker"] = "MX.TO"  # Fix for MX
        return trans_df.sort_values(
            ["datetime", "action", "qty"], ascending=[True, False, True]
        ).reset_index(drop=True)[self.TRANS_COLS]

    def _get_account_info(self, acc_info: Types.AccountInformation):
        """Get the account information."""
        self.account_type = acc_info.accountType if acc_info else None
        self.customer_type = acc_info.customerType if acc_info else None
        self.trading_permissions = acc_info.tradingPermissions if acc_info else None
        self.account_open_date = acc_info.dateOpened if acc_info else None
        self.account_funded_date = acc_info.dateFunded if acc_info else None
        self.base_currency = acc_info.currency if acc_info else None
        self.email = acc_info.primaryEmail if acc_info else None

    def _parse_positions(self, pos_data: List[Types.OpenPosition]) -> dict[Position]:
        """Get position info"""

        positions = {}
        for pos in pos_data:
            ticker = clean_symbol(pos.symbol, pos.currency)

            positions[ticker] = Position(
                ticker=ticker,
                name=pos.description,
                currency=pos.currency,
                category=pos.assetCategory.name,
                side=pos.side.name,
                exchange=pos.listingExchange,
                qty=pos.position,
                book_price=pos.costBasisPrice,
                cost_basis=pos.costBasisMoney,
                market_price=pos.markPrice,
                market_value=pos.positionValue,
                unreal_gain=pos.fifoPnlUnrealized,
                pct_of_nav=pos.percentOfNAV,
                put_call=pos.putCall,
                strike=pos.strike,
                expiry_date=pos.expiry,
                vesting_date=pos.vestingDate,
                last_updated=pos.reportDate,
            )

        # Validate count
        if len(positions) != len(pos_data):
            raise ParseError(
                f"Missing positions -- potential duplicates: {len(pos_data)} vs. {len(positions)}"
            )

        return positions

    def _parse_cash_report(self, cash_reports: List[Types.CashReportCurrency]) -> dict[str, dict]:
        """Get cash report info"""

        # Parse cash reports
        summ = {}
        for cash_report in cash_reports:
            summ[cash_report.currency] = {
                "start": cash_report.startingCash,
                "commissions": cash_report.commissions,
                "deposits": cash_report.deposits,
                "withdrawals": cash_report.withdrawals,
                "transfers": cash_report.accountTransfers,
                "sales": cash_report.netTradesSales,
                "purchases": cash_report.netTradesPurchases,
                "dividends": cash_report.dividends,
                "interest": cash_report.brokerInterest,
                "whtax": cash_report.withholdingTax,
                "inlieu": cash_report.paymentInLieu,
                "fees": cash_report.otherFees,
                "end": cash_report.endingCash,
                "end-settled": cash_report.endingSettledCash,
            }

        # Data processing
        # Create 'CAD' component if it does not exist
        if len(summ) == 1 and "BASE_SUMMARY" in summ:
            summ["CAD"] = summ["BASE_SUMMARY"]

        # Validate cash report
        for curr, curr_vals in summ.items():
            if curr == "BASE_SUMMARY":
                continue

            sum_end = (
                curr_vals["start"]
                + curr_vals["commissions"]
                + curr_vals["deposits"]
                + curr_vals["withdrawals"]
                + curr_vals["transfers"]
                + curr_vals["sales"]
                + curr_vals["purchases"]
                + curr_vals["dividends"]
                + curr_vals["interest"]
                + curr_vals["whtax"]
                + curr_vals["inlieu"]
                + curr_vals["fees"]
            )

            if abs(sum_end - curr_vals["end"]) > self.TOL:
                print(sum_end - curr_vals["end"])
                raise ParseError(
                    f"{curr} Cash report does not add up: {sum_end} != {curr_vals['end']}"
                )

        return summ

    def _parse_trades(self, trade_data: List[Types.Trade]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get trade info"""

        if len(trade_data) == 0:
            return pd.DataFrame(), pd.DataFrame()

        trades = []
        skip_ctr = 0
        for trade in trade_data:
            if isinstance(trade, Types.AssetSummary) or trade.transactionType is None:
                skip_ctr += 1
                continue

            trades.append(
                {
                    "date": trade.dateTime,
                    "trans_type": (trade.transactionType.name if trade.transactionType else None),
                    "buy_sell": (trade.buySell.name if trade.buySell else None),
                    "ticker": clean_symbol(trade.symbol, trade.currency),
                    "qty": trade.quantity,
                    "price": trade.tradePrice,
                    "total": trade.tradeMoney,
                    "fees": trade.ibCommission,
                    "currency": trade.currency,
                    "cost_price": trade.closePrice,
                    "proceeds": trade.proceeds,
                    "open_close": (
                        trade.openCloseIndicator.name if trade.openCloseIndicator else None
                    ),
                    "order_type": (trade.orderType.name if trade.orderType else None),
                    "asset_category": (trade.assetCategory.name if trade.assetCategory else None),
                    "exchange": trade.exchange,
                    "name": trade.description,
                    "net_cash": trade.netCash,
                    "fee_currency": trade.ibCommissionCurrency,
                    "basis": trade.cost,
                    "real_gain": trade.fifoPnlRealized,
                    "put_call": trade.putCall,
                    "strike": trade.strike,
                    "expiry_date": trade.expiry,
                    "order_date": trade.orderTime,
                    "report_date": trade.reportDate,
                    "trade_date": trade.tradeDate,
                    "settle_tgt_date": trade.settleDateTarget,
                    "trans_id": trade.transactionID,
                    "dividend": Types.Decimal(0.0),
                    "action": "",
                }
            )

        # Extract trade and FX datafremes
        all_trades_df = pd.DataFrame(trades)
        all_trades_df.loc[:, "type"] = all_trades_df.loc[:, "buy_sell"]
        trades_df = all_trades_df[all_trades_df["ticker"] != "USD-CAD"].copy()
        fx_df = all_trades_df[all_trades_df["ticker"] == "USD-CAD"].copy()

        # Validate counts
        if len(trade_data) - skip_ctr != len(trades_df) + len(fx_df):
            raise ParseError(
                f"Trades count mismatch: Summary: {len(trade_data) - skip_ctr} vs. transactions: {len(trades_df) + len(fx_df)}"
            )

        # Preprocessing

        ## Update trades_df
        trades_df.loc[:, "fees"] = -trades_df.loc[:, "fees"]

        ## Create cad_fx_df
        cad_fx_df = fx_df.copy()
        cad_fx_df.loc[:, "ticker"] = "CAD.FXT"
        cad_fx_df.loc[:, "fees"] = -cad_fx_df.loc[:, "fees"]
        cad_fx_df.loc[:, "qty"] = Types.Decimal(0.0)

        ## Create usd_fx_df
        usd_fx_df = fx_df.copy()
        usd_fx_df.loc[:, ["ticker", "fees", "currency"]] = ("USD.FXT", Types.Decimal(0.0), "USD")
        usd_fx_df.loc[usd_fx_df["buy_sell"] == "BUY", "total"] = -usd_fx_df.loc[
            usd_fx_df["buy_sell"] == "BUY", "qty"
        ]
        usd_fx_df.loc[usd_fx_df["buy_sell"] == "SELL", "total"] = -usd_fx_df.loc[
            usd_fx_df["buy_sell"] == "SELL", "qty"
        ]
        usd_fx_df.loc[:, "qty"] = Types.Decimal(0.0)
        usd_fx_df.loc[usd_fx_df["buy_sell"] == "BUY", "type"] = "SELL"
        usd_fx_df.loc[usd_fx_df["buy_sell"] == "SELL", "type"] = "BUY"

        return trades_df, pd.concat([cad_fx_df, usd_fx_df], axis=0, ignore_index=True)

    def _parse_corp_actions(self, ca_data: List[Types.CorporateAction]) -> pd.DataFrame:
        """Get corporate action info"""

        if len(ca_data) == 0:
            return pd.DataFrame()

        # Parse corporate actions
        corp_actions = []
        for action in ca_data:
            corp_actions.append(
                {
                    "date": action.dateTime,
                    "ticker": clean_symbol(action.symbol, action.currency),
                    "type": (action.type.name if action.type else None),
                    "qty": action.quantity,
                    "proceeds": action.proceeds,
                    "value": action.value,
                    "asset_category": (action.assetCategory.name if action.assetCategory else None),
                    "currency": action.currency,
                    "description": action.description,
                    "exchange": action.listingExchange,
                    "actionDescription": action.actionDescription,
                    "amount": action.amount,
                    "real_gain": action.fifoPnlRealized,
                    "cap_gain": action.capitalGainsPnl,
                    "dividend": Types.Decimal(0.0),
                    "fees": Types.Decimal(0.0),
                    "action": "CORP",
                    "src_ticker": None,  # TODO: Extract source ticker for CORP Actions
                }
            )
        corp_df = pd.DataFrame(corp_actions)

        # Validate counts
        if len(ca_data) != len(corp_df):
            raise ParseError(
                f"Corporate actions count mismatch: Summary: {len(ca_data)} vs. transactions: {len(corp_df)}"
            )

        # Preprocessing
        corp_df.loc[corp_df["type"].str.endswith("SPLIT"), "action"] = "SPLIT"
        corp_df.loc[:, "ticker"] = corp_df["ticker"].apply(lambda x: x.replace("-OLD", ""))
        corp_df.loc[:, "total"] = -corp_df["proceeds"]

        return corp_df

    def _parse_transfers(self, transfer_data: List[Types.Transfer]) -> pd.DataFrame:
        """Get transfer info"""

        if len(transfer_data) == 0:
            return pd.DataFrame()

        # Parse transfers
        transfers = []
        for tsfr in transfer_data:
            transfers.append(
                {
                    "date": pd.to_datetime(tsfr.date),
                    "type": (tsfr.type.name if tsfr.type else None),
                    "direction": (tsfr.direction.name if tsfr.direction else None),
                    "ticker": clean_symbol(tsfr.symbol, tsfr.currency),
                    "qty": tsfr.quantity,
                    "total": -tsfr.cashTransfer,
                    "transferPrice": tsfr.transferPrice,
                    "positionAmount": tsfr.positionAmount,
                    "currency": tsfr.currency,
                    "asset_category": (tsfr.assetCategory.name if tsfr.assetCategory else None),
                    "exchange": tsfr.listingExchange,
                    "from": tsfr.account,
                    "report_date": tsfr.reportDate,
                    "trans_id": tsfr.transactionID,
                    "action": "TSFR",
                    "fees": Types.Decimal(0.0),
                    "dividend": Types.Decimal(0.0),
                }
            )
        tsfr_df = pd.DataFrame(transfers)

        tsfr_df.loc[tsfr_df["type"] == "ACATS", "ticker"] = (
            tsfr_df.loc[tsfr_df["type"] == "ACATS", "currency"] + ".FXT"
        )
        tsfr_df.loc[tsfr_df["type"] == "ATON", "total"] = Types.Decimal(0.0)

        # Validate counts
        if len(transfer_data) != len(tsfr_df):
            raise ParseError(
                f"Transfers count mismatch: Summary: {len(transfer_data)} vs. transactions: {len(tsfr_df)}"
            )

        return tsfr_df

    def _parse_cash_transactions(
        self, cash_trans_data: List[Types.CashTransaction]
    ) -> pd.DataFrame:
        """Get cash transaction info"""

        if len(cash_trans_data) == 0:
            return pd.DataFrame()

        # Parse cash transactions
        cash_trans = []
        for trans in cash_trans_data:
            cash_trans.append(
                {
                    "date": trans.dateTime,
                    "level": trans.levelOfDetail,
                    "type": (trans.type.name if trans.type else None),
                    "ticker": clean_symbol(trans.symbol, trans.currency),
                    "amount": trans.amount,
                    "currency": trans.currency,
                    "fx_rate": trans.fxRateToBase,
                    "description": trans.description,
                    "report_date": trans.reportDate,
                    "settle_date": trans.settleDate,
                    "asset_category": (trans.assetCategory.name if trans.assetCategory else None),
                    "exchange": trans.listingExchange,
                    "action": "",
                    "qty": Types.Decimal(0.0),
                    "total": Types.Decimal(0.0),
                    "fees": Types.Decimal(0.0),
                    "dividend": Types.Decimal(0.0),
                }
            )

        # Create dataframe
        cash_trans_df = pd.DataFrame(cash_trans)
        skip_ctr = sum(cash_trans_df["level"] == "SUMMARY")
        cash_trans_df = cash_trans_df[cash_trans_df["level"] != "SUMMARY"]

        # Unexpected types
        curr_types = set(cash_trans_df.type.unique().tolist())
        diff = curr_types.difference(self.EXPECTED_CASH_TRANS_TYPES)

        if len(diff) > 0:
            raise ParseError(f"Unexpected cash transaction types: {diff}")

        # Withdrawals / Deposits - Cancellations

        ## Find cancellations
        cash_trans_df.loc[
            (cash_trans_df["type"] == "DEPOSITWITHDRAW")
            & (cash_trans_df["description"] == "CANCELLATION"),
            "type",
        ] = "CANCELLED"

        ## Find matching cancellations
        ## Note: Might need to actually loop this if there is multiple matches

        for i, (date, amt) in cash_trans_df.loc[
            cash_trans_df["type"] == "CANCELLED", ["date", "amount"]
        ].iterrows():

            # Set first match to cancelled
            index = cash_trans_df.loc[
                (cash_trans_df["date"] == date) & (cash_trans_df["amount"] == -amt)
            ].index[0]
            cash_trans_df.loc[index, "type"] = "CANCELLED"

        ## Remove cancelled
        skip_ctr += sum(cash_trans_df["type"] == "CANCELLED")
        cash_trans_df = cash_trans_df[cash_trans_df["type"] != "CANCELLED"]

        # Withdrawals / Deposits
        cash_trans_df.loc[cash_trans_df["type"] == "DEPOSITWITHDRAW", "ticker"] = "CAD"
        cash_trans_df.loc[cash_trans_df["type"] == "DEPOSITWITHDRAW", "total"] = -cash_trans_df.loc[
            cash_trans_df["type"] == "DEPOSITWITHDRAW", "amount"
        ]

        # Dividends / Broker Interest Received / Payment in Lieu of Dividends
        cash_trans_df.loc[
            cash_trans_df["type"].isin(["DIVIDEND", "BROKERINTRCVD", "PAYMENTINLIEU"]), "dividend"
        ] = cash_trans_df.loc[
            cash_trans_df["type"].isin(["DIVIDEND", "BROKERINTRCVD", "PAYMENTINLIEU"]), "amount"
        ]

        # Withholding Tax / Fees / Broker Interest Paid
        cash_trans_df.loc[
            cash_trans_df["type"].isin(["WHTAX", "FEES", "BROKERINTPAID"]), "fees"
        ] = -cash_trans_df.loc[
            cash_trans_df["type"].isin(["WHTAX", "FEES", "BROKERINTPAID"]), "amount"
        ]

        # Fix ticker name -- payments = {currency}.FEES, rewards = {currency}.RWRD
        cash_trans_df.loc[cash_trans_df["type"].isin(["FEES", "BROKERINTPAID"]), "ticker"] = (
            cash_trans_df.loc[cash_trans_df["type"].isin(["FEES", "BROKERINTPAID"]), "currency"]
            + ".FEES"
        )
        cash_trans_df.loc[cash_trans_df["type"] == "BROKERINTRCVD", "ticker"] = (
            cash_trans_df.loc[cash_trans_df["type"] == "BROKERINTRCVD", "currency"] + ".RWRD"
        )

        # Check counts
        if len(cash_trans_data) - skip_ctr != len(cash_trans_df):
            raise ParseError(
                f"Cash transactions count mismatch: Summary: {len(cash_trans_data) - skip_ctr} vs. transactions: {len(cash_trans_df)}"
            )

        return cash_trans_df

    def _validate_data(self) -> None:
        """Validate if sums match cash summary"""

        for curr in ["CAD", "USD"]:

            if curr not in self.summ and len(self.trans_df[self.trans_df["currency"] == curr]) > 0:
                raise ParseError(f"Missing {curr} cash summary")
            elif curr not in self.summ:
                # No curr transactions, skip
                continue

            # Commissions
            tot_comm = -self.trans_df.loc[
                self.trans_df["type"].isin(["BUY", "SELL"]) & (self.trans_df["currency"] == curr),
                "fees",
            ].sum()

            if abs(tot_comm - self.summ[curr]["commissions"]) > self.TOL:
                raise ParseError(
                    f"{curr} Commissions mismatch: Summary: ${self.summ[curr]['commissions']} vs. transactions: ${tot_comm}"
                )

            # Deposits
            tot_deps = -self.trans_df.loc[
                (self.trans_df["type"] == "DEPOSITWITHDRAW")
                & (self.trans_df["total"] < 0)
                & (self.trans_df["currency"] == curr),
                "total",
            ].sum()

            if abs(tot_deps - self.summ[curr]["deposits"]) > self.TOL:
                raise ParseError(
                    f"{curr} Deposits mismatch: Summary: ${self.summ[curr]['deposits']} vs. transactions: ${tot_deps}"
                )

            # Withdrawals
            tot_wds = -self.trans_df.loc[
                (self.trans_df["type"] == "DEPOSITWITHDRAW")
                & (self.trans_df["total"] > 0)
                & (self.trans_df["currency"] == curr),
                "total",
            ].sum()

            if abs(tot_wds - self.summ[curr]["withdrawals"]) > self.TOL:
                raise ParseError(
                    f"{curr} Withdrawals mismatch: Summary: ${self.summ[curr]['withdrawals']} vs. transactions: ${tot_wds}"
                )

            # Dividends
            tot_div = self.trans_df.loc[
                (self.trans_df["type"] == "DIVIDEND") & (self.trans_df["currency"] == curr),
                "dividend",
            ].sum()

            if abs(tot_div - self.summ[curr]["dividends"]) > self.TOL:
                raise ParseError(
                    f"{curr} Dividends mismatch: Summary: ${self.summ[curr]['dividends']} vs. transactions: ${tot_div}"
                )

            # Trades (Sales)
            tot_sales = -self.trans_df.loc[
                (self.trans_df["type"].isin(["SELL", "MERGER"]))
                & (self.trans_df["currency"] == curr),
                "total",
            ].sum()

            if abs(tot_sales - self.summ[curr]["sales"]) > self.TOL:
                raise ParseError(
                    f"{curr} Trades (Sales) mismatch: Summary: ${self.summ[curr]['sales']} vs. transactions: ${tot_sales}"
                )

            # Trades (Purchases)
            tot_purchs = -self.trans_df.loc[
                (self.trans_df["type"] == "BUY") & (self.trans_df["currency"] == curr),
                "total",
            ].sum()

            if abs(tot_purchs - self.summ[curr]["purchases"]) > self.TOL:
                raise ParseError(
                    f"{curr} Trades (Purchases) mismatch: Summary: ${self.summ[curr]['purchases']} vs. transactions: ${tot_purchs}"
                )

            # Withholding Tax
            tot_whtax = -self.trans_df.loc[
                (self.trans_df["type"] == "WHTAX") & (self.trans_df["currency"] == curr), "fees"
            ].sum()

            if abs(tot_whtax - self.summ[curr]["whtax"]) > self.TOL:
                raise ParseError(
                    f"{curr} Withholding Tax mismatch: Summary: ${self.summ[curr]['whtax']} vs. transactions: ${tot_whtax}"
                )

            # Interest
            tot_int = (
                self.trans_df.loc[
                    (self.trans_df["type"] == "BROKERINTRCVD")
                    & (self.trans_df["currency"] == curr),
                    "dividend",
                ].sum()
                - self.trans_df.loc[
                    (self.trans_df["type"] == "BROKERINTPAID")
                    & (self.trans_df["currency"] == curr),
                    "fees",
                ].sum()
            )

            if abs(tot_int - self.summ[curr]["interest"]) > self.TOL:
                raise ParseError(
                    f"{curr} Interest mismatch: Summary: ${self.summ[curr]['interest']} vs. transactions: ${tot_int}"
                )

            # Fees
            tot_fees = -self.trans_df.loc[
                (self.trans_df["type"] == "FEES") & (self.trans_df["currency"] == curr), "fees"
            ].sum()

            if abs(tot_fees - self.summ[curr]["fees"]) > self.TOL:
                raise ParseError(
                    f"{curr} Fees mismatch: Summary: ${self.summ[curr]['fees']} vs. transactions: ${tot_fees}"
                )

            # In lieu of dividends
            tot_inlieu = self.trans_df.loc[
                (self.trans_df["type"] == "PAYMENTINLIEU") & (self.trans_df["currency"] == curr),
                "dividend",
            ].sum()

            if abs(tot_inlieu - self.summ[curr]["inlieu"]) > self.TOL:
                raise ParseError(
                    f"{curr} In lieu of dividends mismatch: Summary: ${self.summ[curr]['inlieu']} vs. transactions: ${tot_inlieu}"
                )

            # Transfers
            tot_tsfrs = -self.trans_df.loc[
                (self.trans_df["type"].isin(["ATON", "ACATS"]))
                & (self.trans_df["currency"] == curr),
                "total",
            ].sum()

            if abs(tot_tsfrs - self.summ[curr]["transfers"]) > self.TOL:
                raise ParseError(
                    f"{curr} Transfers mismatch: Summary: ${self.summ[curr]['transfers']} vs. transactions: ${tot_tsfrs}"
                )

            # Ending Cash
            tot_endcash = (
                self.summ[curr]["start"]
                + tot_comm
                + tot_deps
                + tot_wds
                + tot_div
                + tot_sales
                + tot_purchs
                + tot_whtax
                + tot_int
                + tot_fees
                + tot_inlieu
                + tot_tsfrs
            )

            if abs(self.summ[curr]["end"] - tot_endcash) > self.TOL:
                raise ValueError(
                    f"{curr} Ending cash mismatch: Summary: ${self.summ[curr]['end']} vs. transactions: ${tot_endcash}"
                )
