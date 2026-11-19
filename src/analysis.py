# -*- coding: utf-8 -*-
"""
Step 1 | 电商用户购买行为分析（指标层）
-------------------------------------------------------------------
从 5 张原始表出发，产出：
  1) dashboard/assets/js/data.js   看板所需的全部指标数据
  2) reports/metrics.txt           所有关键数值（写报告/简历引用）

分析框架：
  A. 经营大盘   —— 净GMV / 客单价 / 转化率 / 复购率 的月度走势
  B. 转化漏斗   —— 浏览→加购→下单→支付，按渠道/设备拆解找漏点
  C. 用户分层   —— RFM 八分群，看"人群占比 vs GMV 占比"的错配
  D. 新老客     —— 新客贡献、次月留存、月度 Cohort 留存
  E. 品类结构   —— 份额×份额增速四象限 + 毛利/退款率双维
  F. 时空规律   —— 周×小时下单热力、价格带分布、大促贡献

运行：python src/analysis.py
"""
import os
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT_JS = os.path.join(ROOT, "dashboard", "assets", "js", "data.js")
OUT_TXT = os.path.join(ROOT, "reports", "metrics.txt")

CATS = ["女装", "美妆个护", "食品生鲜", "3C数码", "家居日用", "母婴玩具", "运动户外"]

def load():
    orders = pd.read_csv(os.path.join(RAW, "orders.csv"), parse_dates=["order_date"])
    items = pd.read_csv(os.path.join(RAW, "order_items.csv"))
    sessions = pd.read_csv(os.path.join(RAW, "sessions.csv"), parse_dates=["dt"])
    users = pd.read_csv(os.path.join(RAW, "users.csv"), parse_dates=["reg_date"])
    orders["net_gmv"] = orders["pay_amount"] - orders["refund_amount"]
    orders["month"] = orders["order_date"].dt.strftime("%Y-%m")
    sessions["month"] = sessions["dt"].dt.strftime("%Y-%m")
    return orders, items, sessions, users


def promo_label(d):
    m, dd = d.month, d.day
    if (m, dd) >= (11, 1) and (m, dd) <= (11, 11): return "双11"
    if (m == 5 and dd >= 26) or (m == 6 and dd <= 18): return "618"
    if m == 12 and 9 <= dd <= 12: return "双12"
    if m == 3 and 5 <= dd <= 8: return "38节"
    if m == 9 and 5 <= dd <= 9: return "99节"
    if (m == 1 and dd >= 25) or (m == 2 and dd <= 4): return "年货节"
    if m == 12 and dd >= 24: return "双旦"
    return "日常"


# ============================================================ A. 经营大盘
def kpi_and_trend(orders, items, sessions):
    def block(df):
        return dict(
            net_gmv=float(df["net_gmv"].sum()),
            orders=int(len(df)),
            buyers=int(df["user_id"].nunique()),
            aov=float(df["pay_amount"].mean()),
        )
    h1 = block(orders[orders["order_date"].dt.month <= 6])
    h2 = block(orders[orders["order_date"].dt.month > 6])
    sess_h1 = sessions[sessions["dt"].dt.month <= 6]
    sess_h2 = sessions[sessions["dt"].dt.month > 6]
    cvr_h1 = (sess_h1["max_stage"] == 4).mean()
    cvr_h2 = (sess_h2["max_stage"] == 4).mean()

    rep = orders.groupby("user_id").size()
    repurchase = float((rep >= 2).mean())
    refund_amt_rate = float(orders["refund_amount"].sum() / orders["pay_amount"].sum())
    refund_ord_rate = float(orders["refund_flag"].mean())
    it = items.merge(orders[["order_id"]], on="order_id")
    gross_margin = float(1 - (it["amount"] * it["cost_rate"]).sum() / it["amount"].sum())

    kpis = [
        dict(label="净GMV", value=round(orders["net_gmv"].sum() / 10000, 1), unit="万元",
             delta=round(h2["net_gmv"] / h1["net_gmv"] - 1, 4), note="实付-退款", fmt="num1"),
        dict(label="订单量", value=int(len(orders)), unit="笔",
             delta=round(h2["orders"] / h1["orders"] - 1, 4), note="全年已支付", fmt="int"),
        dict(label="购买用户数", value=int(orders["user_id"].nunique()), unit="人",
             delta=round(h2["buyers"] / h1["buyers"] - 1, 4), note="全年至少1单", fmt="int"),
        dict(label="客单价 AOV", value=round(float(orders["pay_amount"].mean()), 1), unit="元",
             delta=round(h2["aov"] / h1["aov"] - 1, 4), note="实付金额/订单", fmt="num1"),
        dict(label="支付转化率", value=round(float((sessions["max_stage"] == 4).mean()) * 100, 2), unit="%",
             delta=round(cvr_h2 / cvr_h1 - 1, 4), note="支付会话/总访问", fmt="num2"),
        dict(label="复购率", value=round(repurchase * 100, 1), unit="%",
             delta=None, note="下单≥2次的买家占比", fmt="num1"),
        dict(label="退款率", value=round(refund_ord_rate * 100, 2), unit="%",
             delta=None, note="退款订单/支付订单", fmt="num2"),
        dict(label="毛利率", value=round(gross_margin * 100, 1), unit="%",
             delta=None, note="按商品吊牌成本估算", fmt="num1"),
    ]

    # 月度趋势
    m = orders.groupby("month").agg(net_gmv=("net_gmv", "sum"),
                                    orders=("order_id", "size"),
                                    buyers=("user_id", "nunique"),
                                    aov=("pay_amount", "mean")).reset_index()
    sv = sessions.groupby("month").apply(
        lambda g: (g["max_stage"] == 4).mean(), include_groups=False).rename("cvr").reset_index()
    sv.columns = ["month", "cvr"]
    m = m.merge(sv, on="month")
    monthly = dict(
        months=[x[2:] for x in m["month"]],
        net_gmv=[round(v / 10000, 1) for v in m["net_gmv"]],
        orders=[int(v) for v in m["orders"]],
        buyers=[int(v) for v in m["buyers"]],
        aov=[round(v, 1) for v in m["aov"]],
        cvr=[round(v * 100, 2) for v in m["cvr"]],
    )
    return kpis, monthly


# ============================================================ B. 转化漏斗
def funnel(sessions, orders):
    n = len(sessions)
    c1 = n
    c2 = int((sessions["max_stage"] >= 2).sum())
    c3 = int((sessions["max_stage"] >= 3).sum())
    c4 = int((sessions["max_stage"] >= 4).sum())
    steps = [
        dict(name="浏览", value=c1, rate=100.0),
        dict(name="加购", value=c2, rate=round(c2 / c1 * 100, 2)),
        dict(name="下单", value=c3, rate=round(c3 / c1 * 100, 2)),
        dict(name="支付", value=c4, rate=round(c4 / c1 * 100, 2)),
    ]
    steps[1]["step"] = round(c2 / c1 * 100, 1)
    steps[2]["step"] = round(c3 / c2 * 100, 1)
    steps[3]["step"] = round(c4 / c3 * 100, 1)

    by_channel = []
    for ch, g in sessions.groupby("channel"):
        n2 = len(g)
        o = orders[orders["channel"] == ch]
        by_channel.append(dict(
            name=ch,
            sessions=int(n2),
            cart_rate=round((g["max_stage"] >= 2).mean() * 100, 2),
            order_rate=round((g["max_stage"] >= 3).mean() * 100, 2),
            cvr=round((g["max_stage"] == 4).mean() * 100, 2),
            orders=int(len(o)),
            net_gmv=round(float(o["net_gmv"].sum()) / 10000, 1),
            gmv_share=round(float(o["net_gmv"].sum()) / orders["net_gmv"].sum() * 100, 2),
            rps=round(float(o["net_gmv"].sum()) / n2, 2),
            aov=round(float(o["pay_amount"].mean()), 1),
            refund_rate=round(float(o["refund_flag"].mean()) * 100, 2),
            new_ratio=round(float(o["is_new_buyer"].mean()) * 100, 1),
        ))
    by_channel.sort(key=lambda x: -x["cvr"])

    by_device = []
    for dv, g in sessions.groupby("device"):
        o = orders[orders["device"] == dv]
        by_device.append(dict(name=dv,
                              sessions=int(len(g)),
                              cvr=round((g["max_stage"] == 4).mean() * 100, 2),
                              aov=round(float(o["pay_amount"].mean()), 1),
                              net_gmv=round(float(o["net_gmv"].sum()) / 10000, 1)))
    by_device.sort(key=lambda x: -x["cvr"])
    return steps, by_channel, by_device


# ============================================================ C. RFM 分层
def rfm(orders):
    snap = pd.Timestamp("2025-12-31")
    g = orders.groupby("user_id").agg(
        recency=("order_date", lambda s: (snap - s.max()).days),
        frequency=("order_id", "size"),
        monetary=("net_gmv", "sum"),
        aov=("pay_amount", "mean"),
        last=("order_date", "max"),
    )
    g["r"] = pd.qcut(g["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    g["f"] = pd.qcut(g["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    g["m"] = pd.qcut(g["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    def name_of(r, f, m):
        rh, fh, mh = r >= 4, f >= 4, m >= 4
        if mh:
            if rh and fh:   return "重要价值客户"
            if fh:          return "重要保持客户"
            if rh:          return "重要发展客户"
            return "重要挽留客户"
        if rh and fh:       return "一般价值客户"
        if fh:              return "一般保持客户"
        if rh:              return "一般发展客户"
        return "一般挽留客户"

    g["segment"] = [name_of(r, f, m) for r, f, m in zip(g["r"], g["f"], g["m"])]
    total_gmv = g["monetary"].sum()
    total_u = len(g)

    order = ["重要价值客户", "重要保持客户", "重要发展客户", "重要挽留客户",
             "一般价值客户", "一般保持客户", "一般发展客户", "一般挽留客户"]
    order = [x for x in order if x in set(g["segment"])]
    segs = []
    for s in order:
        sub = g[g["segment"] == s]
        segs.append(dict(
            name=s,
            users=int(len(sub)),
            user_share=round(len(sub) / total_u * 100, 2),
            gmv=round(float(sub["monetary"].sum()) / 10000, 1),
            gmv_share=round(float(sub["monetary"].sum()) / total_gmv * 100, 2),
            aov=round(float(sub["aov"].mean()), 1),
            freq=round(float(sub["frequency"].mean()), 2),
            recency=round(float(sub["recency"].mean()), 0),
        ))
    # 帕累托
    srt = g["monetary"].sort_values(ascending=False).to_numpy()
    cum = np.cumsum(srt) / srt.sum()
    top20 = float(cum[int(len(srt) * 0.2) - 1])
    return segs, dict(top20_gmv_share=round(top20 * 100, 1), buyers=total_u)


# ============================================================ D. 新老客 & 留存
def new_old(orders):
    m = orders.groupby(["month", "is_new_buyer"]).agg(
        gmv=("net_gmv", "sum"), orders=("order_id", "size")).reset_index()
    piv = m.pivot(index="month", columns="is_new_buyer", values="gmv").fillna(0)
    piv.columns = ["old_gmv" if c == 0 else "new_gmv" for c in piv.columns]
    po = m.pivot(index="month", columns="is_new_buyer", values="orders").fillna(0)
    po.columns = ["old_orders" if c == 0 else "new_orders" for c in po.columns]
    df = piv.join(po).reset_index()

    # 新客次月留存
    first = orders[orders["is_new_buyer"] == 1][["user_id", "month"]].rename(columns={"month": "c0"})
    act = orders.groupby("user_id")["month"].apply(set)
    rows = []
    months_all = set(orders["month"].unique())
    for mo in sorted(orders["month"].unique()):
        cohort = first[first["c0"] == mo]["user_id"]
        nxt = (pd.Period(mo) + 1).strftime("%Y-%m")
        if len(cohort) == 0 or nxt not in months_all:
            continue
        kept = sum(1 for u in cohort if nxt in act.get(u, set()))
        rows.append(dict(month=mo[2:], size=int(len(cohort)), m1_rate=round(kept / len(cohort) * 100, 2)))

    monthly = []
    for _, r in df.iterrows():
        tot = r["new_gmv"] + r["old_gmv"]
        monthly.append(dict(
            month=r["month"][2:],
            new_gmv=round(r["new_gmv"] / 10000, 1),
            old_gmv=round(r["old_gmv"] / 10000, 1),
            new_ratio=round(r["new_gmv"] / tot * 100, 1) if tot else 0,
            new_orders=int(r["new_orders"]),
            old_orders=int(r["old_orders"]),
        ))
    return monthly, rows


def cohort(orders):
    first = orders.groupby("user_id")["month"].min().rename("cohort")
    d = orders.merge(first, on="user_id")
    d["idx"] = (pd.PeriodIndex(d["month"], freq="M") -
                pd.PeriodIndex(d["cohort"], freq="M")).map(lambda p: p.n)
    piv = d.groupby(["cohort", "idx"])["user_id"].nunique().unstack(fill_value=0)
    size = d.groupby("cohort")["user_id"].nunique()
    out = []
    max_idx = 5
    for co in sorted(piv.index):
        if co > "2025-09":
            continue
        row = {"cohort": co[2:], "size": int(size[co]), "rates": []}
        avail = (pd.Period("2025-12", freq="M") - pd.Period(co, freq="M")).n
        for i in range(0, max_idx + 1):
            if i > avail or i not in piv.columns:
                row["rates"].append(None)
            else:
                row["rates"].append(round(piv.loc[co, i] / size[co] * 100, 1))
        out.append(row)
    return out


# ============================================================ E. 品类结构
def category(orders, items):
    it = items.merge(orders[["order_id", "month", "net_gmv"]], on="order_id", how="left")
    it["half"] = np.where(it["month"].str[5:7].astype(int) <= 6, "H1", "H2")
    g = it.groupby("category_l1").agg(
        amount=("amount", "sum"),
        orders=("order_id", "nunique"),
        cost=("amount", lambda s: (s * it.loc[s.index, "cost_rate"]).sum()),
    )
    # 品类 GMV 用订单主品类口径，与整体 net_gmv 对齐
    main = orders.groupby("main_category").agg(
        net_gmv=("net_gmv", "sum"), orders=("order_id", "size"),
        aov=("pay_amount", "mean"), refund_rate=("refund_flag", "mean")).reset_index()
    share_h = it.groupby(["category_l1", "half"])["amount"].sum().unstack()
    share_h = share_h.div(share_h.sum(axis=0), axis=1)
    share_growth = (share_h["H2"] / share_h["H1"] - 1)

    total = main["net_gmv"].sum()
    rows = []
    for _, r in main.iterrows():
        c = r["main_category"]
        cost_amt = it[it["category_l1"] == c]
        gm = 1 - float((cost_amt["amount"] * cost_amt["cost_rate"]).sum() / cost_amt["amount"].sum())
        rows.append(dict(
            name=c,
            gmv=round(float(r["net_gmv"]) / 10000, 1),
            share=round(float(r["net_gmv"]) / total * 100, 2),
            orders=int(r["orders"]),
            aov=round(float(r["aov"]), 1),
            growth=round(float(share_growth.get(c, 0)) * 100, 1),
            margin=round(gm * 100, 1),
            refund=round(float(r["refund_rate"]) * 100, 2),
        ))
    rows.sort(key=lambda x: -x["gmv"])
    return rows


# ============================================================ F. 时空 / 价格带 / 大促
def heatmap(orders):
    d = orders.copy()
    d["dow"] = d["order_date"].dt.day_of_week  # 0=周一
    h = d.groupby(["dow", "order_hour"]).size().unstack(fill_value=0)
    data = []
    tot = len(d)
    for dw in range(7):
        for hr in range(24):
            v = int(h.loc[dw, hr]) if (dw in h.index and hr in h.columns) else 0
            data.append([hr, dw, round(v / tot * 100, 3)])
    peak = max(data, key=lambda x: x[2])
    return data, dict(hour=int(peak[0]), dow=int(peak[1]), share=float(peak[2]))


def price_band(orders):
    bins = [0, 50, 100, 200, 500, 1000, 1e9]
    labels = ["0-50", "50-100", "100-200", "200-500", "500-1000", "1000+"]
    b = pd.cut(orders["pay_amount"], bins=bins, labels=labels, right=False)
    g = orders.groupby(b, observed=False).agg(orders=("order_id", "size"), gmv=("net_gmv", "sum"))
    tot = g["orders"].sum()
    gt = g["gmv"].sum()
    out = []
    for lb in labels:
        if lb in g.index:
            out.append(dict(band=lb, orders=int(g.loc[lb, "orders"]),
                            order_share=round(g.loc[lb, "orders"] / tot * 100, 1),
                            gmv=round(float(g.loc[lb, "gmv"]) / 10000, 1),
                            gmv_share=round(float(g.loc[lb, "gmv"]) / gt * 100, 1)))
    return out


def promo(orders, sessions):
    orders = orders.copy()
    orders["promo"] = [promo_label(d) for d in orders["order_date"]]
    sessions = sessions.copy()
    sessions["promo"] = [promo_label(d) for d in sessions["dt"]]
    sv = sessions.groupby("promo").apply(lambda g: (g["max_stage"] == 4).mean(),
                                        include_groups=False).rename("cvr")
    og = orders.groupby("promo").agg(days=("order_date", "nunique"), orders=("order_id", "size"),
                                     net_gmv=("net_gmv", "sum"), aov=("pay_amount", "mean"),
                                     disc=("discount_rate", "mean"))
    og = og.join(sv).reset_index()
    og["gmv_per_day"] = og["net_gmv"] / og["days"]
    base = og.loc[og["promo"] == "日常", "gmv_per_day"].iloc[0]
    out = []
    for _, r in og.sort_values("gmv_per_day", ascending=False).iterrows():
        out.append(dict(name=r["promo"], days=int(r["days"]), orders=int(r["orders"]),
                        net_gmv=round(float(r["net_gmv"]) / 10000, 1),
                        gmv_per_day=round(float(r["gmv_per_day"]) / 10000, 2),
                        lift=round(float(r["gmv_per_day"] / base), 2),
                        aov=round(float(r["aov"]), 1),
                        cvr=round(float(r["cvr"]) * 100, 2),
                        disc=round(float(r["disc"]) * 100, 1)))
    return out


def main():
    orders, items, sessions, users = load()
    kpis, monthly = kpi_and_trend(orders, items, sessions)
    steps, by_channel, by_device = funnel(sessions, orders)
    segs, pareto = rfm(orders)
    no_monthly, new_ret = new_old(orders)
    coh = cohort(orders)
    cats = category(orders, items)
    hm, peak = heatmap(orders)
    pb = price_band(orders)
    pr = promo(orders, sessions)

    data = dict(
        meta=dict(period="2025-01-01 ~ 2025-12-31",
                  users_total=int(len(users)),
                  buyers=int(orders["user_id"].nunique()),
                  sessions=int(len(sessions)),
                  orders=int(len(orders)),
                  updated="2025-12-31"),
        kpi=kpis, monthly=monthly, funnel=steps, funnel_channel=by_channel,
        funnel_device=by_device, rfm=segs, pareto=pareto,
        new_old=no_monthly, new_retention=new_ret, cohort=coh,
        category=cats, heatmap=hm, heatmap_peak=peak,
        price_band=pb, promo=pr,
    )

    os.makedirs(os.path.dirname(OUT_JS), exist_ok=True)
    with open(OUT_JS, "w", encoding="utf-8") as f:
        f.write("window.DASHBOARD_DATA = ")
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write(";\n")

    # ---------------- 数值摘要（供报告引用）
    L = []
    P = L.append
    P("=" * 68)
    P("电商用户购买行为分析 · 关键指标")
    P("=" * 68)
    for k in kpis:
        d = "" if k["delta"] is None else f"  (H2环比 {k['delta']*100:+.1f}%)"
        P(f"{k['label']:<10}{k['value']:>12} {k['unit']}{d}")
    P("")
    P("-- 漏斗 --")
    for s in steps:
        st = f"  环节转化 {s['step']}%" if "step" in s else ""
        P(f"{s['name']:<6}{s['value']:>8}  占浏览 {s['rate']:>6}%{st}")
    P("")
    P("-- 渠道 --")
    for c in by_channel:
        P(f"{c['name']:<7}会话{c['sessions']:>7} CVR{c['cvr']:>6}%  净GMV{c['net_gmv']:>7}万({c['gmv_share']:>5}%) "
          f"每会话{c['rps']:>6}元 AOV{c['aov']:>7} 退款{c['refund_rate']:>5}%")
    P("")
    P("-- RFM --")
    for s in segs:
        P(f"{s['name']:<7}人数{s['users']:>6}({s['user_share']:>5}%)  GMV{s['gmv']:>8}万({s['gmv_share']:>5}%)  "
          f"AOV{s['aov']:>7}  频次{s['freq']:>5}  R{s['recency']:>4}天")
    P(f"头部20%用户贡献净GMV: {pareto['top20_gmv_share']}%")
    P("")
    P("-- 品类 --")
    for c in cats:
        P(f"{c['name']:<6}净GMV{c['gmv']:>8}万 份额{c['share']:>5}% 增速{c['growth']:>6}% "
          f"毛利{c['margin']:>5}% 退款{c['refund']:>5}% AOV{c['aov']:>7}")
    P("")
    P("-- 大促 --")
    for p in pr:
        P(f"{p['name']:<5}天数{p['days']:>3} 单均日产{p['gmv_per_day']:>7}万 倍数{p['lift']:>5} "
          f"AOV{p['aov']:>7} CVR{p['cvr']:>6}% 折扣{p['disc']:>5}%")
    P("")
    P("-- 价格带 --")
    for b in pb:
        P(f"{b['band']:<10}订单占比{b['order_share']:>5}%  GMV占比{b['gmv_share']:>5}%")
    P("")
    P(f"下单峰值时段: 周{peak['dow']+1} {peak['hour']}时")
    P("-- 新客次月留存 --")
    for r in new_ret:
        P(f"{r['month']} 新客{r['size']:>5}人  次月留存{r['m1_rate']:>5}%")

    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\n看板数据 -> {OUT_JS}")


if __name__ == "__main__":
    main()
