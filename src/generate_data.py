# -*- coding: utf-8 -*-
"""
Step 0 | 模拟电商平台原始数据（可复现）
-------------------------------------------------------------------
产出 5 张表，模拟一个中等规模电商平台的 2025 全年数据：
  users.csv        用户维表
  products.csv    商品维表
  sessions.csv    会话表（含转化漏斗所处层级）
  orders.csv      订单表（已支付）
  order_items.csv 订单明细表

设计要点（让数据"长得像真的"，后面的分析才有意义）：
1. 用户存在潜在价值分层（重度/中度/轻度/一次性），影响访问频次与转化意愿
2. 流量存在明显季节性：38、618、99、双11、双12、年货节、春节低谷
3. 渠道特性差异：直播拉加购但支付率低，私域支付率高但流量小，自然搜索意图最强
4. 品类有价格带与退款率差异（服饰退款高、3C 退款低）
5. 订单来自"会话转化"，因此漏斗数据与订单数据天然自洽

运行：python src/generate_data.py
"""
import os
import numpy as np
import pandas as pd

SEED = 2025
rng = np.random.default_rng(SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW, exist_ok=True)

START, END = "2025-01-01", "2025-12-31"
N_USERS = 60_000
DAILY_BASE_SESSIONS = 1_500

# ---------------------------------------------------------------- 1. 用户维表
# 潜在价值分层（模拟时使用，不落库，避免"作弊"）
SEGMENTS = ["重度", "中度", "轻度", "一次性"]
SEG_P = [0.07, 0.28, 0.45, 0.20]
# 访问权重：重度用户贡献绝大多数会话
SEG_WEIGHT = {"重度": 7.0, "中度": 2.1, "轻度": 0.75, "一次性": 0.28}
# 转化意愿乘数
SEG_INTENT = {"重度": 1.75, "中度": 1.05, "轻度": 0.70, "一次性": 0.45}

CHANNELS = ["自然搜索", "信息流广告", "社交裂变", "直播引流", "私域社群"]
CHANNEL_P = [0.26, 0.28, 0.16, 0.18, 0.12]
CITY_TIERS = ["一线", "新一线", "二线", "三线及以下"]
CITY_P = [0.14, 0.26, 0.32, 0.28]

seg = rng.choice(SEGMENTS, N_USERS, p=SEG_P)
# 分层与消费能力弱相关：重度用户更可能来自高线
city_p_matrix = np.array([
    [0.30, 0.34, 0.24, 0.12],   # 重度
    [0.12, 0.28, 0.34, 0.26],   # 中度
    [0.08, 0.23, 0.33, 0.36],   # 轻度
    [0.10, 0.22, 0.32, 0.36],   # 一次性
])
seg_idx = np.array([SEGMENTS.index(s) for s in seg])
city = np.array([rng.choice(CITY_TIERS, p=city_p_matrix[i]) for i in seg_idx])

reg_start = pd.Timestamp("2024-01-01")
reg_days = (pd.Timestamp(END) - reg_start).days
reg_date = reg_start + pd.to_timedelta(rng.integers(0, reg_days, N_USERS), unit="D")

member_level = np.where(
    seg == "重度", "黑卡",
    np.where(seg == "中度", rng.choice(["金牌", "银牌"], N_USERS, p=[0.45, 0.55]),
             np.where(seg == "轻度", rng.choice(["银牌", "普通"], N_USERS, p=[0.35, 0.65]), "普通")))

users = pd.DataFrame({
    "user_id": [f"U{i:06d}" for i in range(1, N_USERS + 1)],
    "reg_date": reg_date.strftime("%Y-%m-%d"),
    "acq_channel": rng.choice(CHANNELS, N_USERS, p=CHANNEL_P),
    "city_tier": city,
    "gender": rng.choice(["女", "男", "未知"], N_USERS, p=[0.62, 0.31, 0.07]),
    "age_group": rng.choice(["18-24", "25-30", "31-40", "41+"], N_USERS, p=[0.22, 0.34, 0.29, 0.15]),
    "member_level": member_level,
})

# ---------------------------------------------------------------- 2. 商品维表
CATS = {
    "女装":        {"n": 220, "median": 159, "cost": 0.42, "refund": 0.145},
    "美妆个护":     {"n": 150, "median": 129, "cost": 0.38, "refund": 0.062},
    "食品生鲜":     {"n": 130, "median": 59,  "cost": 0.66, "refund": 0.048},
    "3C数码":      {"n": 90,  "median": 699, "cost": 0.78, "refund": 0.035},
    "家居日用":     {"n": 120, "median": 109, "cost": 0.55, "refund": 0.055},
    "母婴玩具":     {"n": 100, "median": 149, "cost": 0.52, "refund": 0.072},
    "运动户外":     {"n": 90,  "median": 239, "cost": 0.48, "refund": 0.088},
}
SUB = {
    "女装": ["连衣裙", "上衣T恤", "裤装", "外套", "内衣"],
    "美妆个护": ["面部护肤", "彩妆", "洗发护发", "身体护理", "口腔护理"],
    "食品生鲜": ["休闲零食", "粮油调味", "乳品饮料", "水果生鲜", "速食"],
    "3C数码": ["手机通讯", "电脑办公", "影音耳机", "智能设备", "配件"],
    "家居日用": ["家纺布艺", "收纳清洁", "厨房用具", "家装软饰"],
    "母婴玩具": ["奶粉辅食", "纸尿裤", "童装童鞋", "益智玩具"],
    "运动户外": ["运动鞋服", "健身器材", "户外装备", "骑行"],
}

rows = []
pid = 1
cat_names = list(CATS.keys())
cat_share = np.array([0.26, 0.19, 0.21, 0.07, 0.13, 0.09, 0.05])  # 品类结构
cat_p_adj = cat_share / cat_share.sum()
cat_weight = {}   # 每个品类内部的商品热度权重
for ci, c in enumerate(cat_names):
    cfg = CATS[c]
    n = cfg["n"]
    # 帕累托结构：少数爆款吃掉多数流量
    popularity = rng.pareto(1.6, n) + 1
    popularity = popularity / popularity.sum()
    cat_weight[c] = popularity
    subs = SUB[c]
    sub_idx = rng.choice(len(subs), n, p=np.abs(rng.dirichlet(np.ones(len(subs)) * 2.5)))
    for i in range(n):
        price = max(9.9, float(np.round(cfg["median"] * rng.lognormal(0, 0.55), 1)))
        if c == "3C数码":
            price = max(39.0, float(np.round(price, 0)))
        rows.append({
            "product_id": f"P{pid:05d}",
            "category_l1": c,
            "category_l2": subs[sub_idx[i]],
            "list_price": price,
            "cost_rate": round(min(0.95, max(0.25, rng.normal(cfg["cost"], 0.05))), 3),
            "launch_date": (pd.Timestamp("2025-01-01") -
                            pd.to_timedelta(int(rng.integers(0, 540)), unit="D")).strftime("%Y-%m-%d"),
        })
        pid += 1

products = pd.DataFrame(rows)
prod_by_cat = {c: products.index[products["category_l1"] == c].to_numpy() for c in cat_names}
prod_price = products["list_price"].to_numpy()
prod_cost = products["cost_rate"].to_numpy()
prod_cat = products["category_l1"].to_numpy()
cat_refund = np.array([CATS[c]["refund"] for c in cat_names])

# ---------------------------------------------------------------- 3. 会话表
dates = pd.date_range(START, END, freq="D")

def day_factor(d):
    """日期流量系数：大促脉冲 + 周末 + 春节低谷"""
    m, dd = d.month, d.day
    f = 1.0
    if (m, dd) >= (1, 28) and (m, dd) <= (2, 4):        # 春节
        f *= 0.55
    if m == 3 and 5 <= dd <= 8:                          # 38 女王节
        f *= 1.55
    if (m == 5 and dd >= 26) or (m == 6 and dd <= 18):   # 618 长尾
        f *= 1.9 + (0.9 if (m == 6 and dd >= 15) else 0)
    if m == 9 and 5 <= dd <= 9:                          # 99 划算节
        f *= 1.45
    if (m == 10 and dd >= 21) or (m == 11 and dd <= 11): # 双11 预售+爆发
        f *= 1.75 + (1.6 if (m == 11 and dd >= 9) else 0)
    if m == 12 and 9 <= dd <= 12:                        # 双12
        f *= 1.55
    if m == 12 and dd >= 24:                             # 双旦
        f *= 1.30
    if d.day_of_week >= 5:
        f *= 1.14
    return f

day_counts = np.array([int(DAILY_BASE_SESSIONS * day_factor(d) * rng.normal(1.0, 0.06)) for d in dates])
day_counts = np.maximum(day_counts, 200)

# 用户在当天的访问权重（重度>中度>轻度；注册后才可能访问）
base_w = np.array([SEG_WEIGHT[s] for s in seg]) * rng.lognormal(0, 0.45, N_USERS)
reg_ord = users["reg_date"].to_numpy()
reg_ts = pd.to_datetime(reg_ord)

HOUR_P = np.array([0.006, 0.004, 0.003, 0.003, 0.004, 0.007, 0.012, 0.022, 0.036, 0.050,
                   0.062, 0.062, 0.052, 0.050, 0.054, 0.054, 0.050, 0.045, 0.043, 0.048,
                   0.058, 0.068, 0.062, 0.045, 0.030, 0.018, 0.011, 0.006])
HOUR_P = HOUR_P / HOUR_P.sum()

# 渠道 × 环节转化率参数（这是"业务规律"的核心来源）
CH_CONV = {
    #            加购率  下单率(条件) 支付率(条件)
    "自然搜索": (0.178, 0.300, 0.800),
    "信息流广告": (0.082, 0.255, 0.640),
    "社交裂变": (0.118, 0.275, 0.715),
    "直播引流": (0.198, 0.300, 0.595),
    "私域社群": (0.152, 0.355, 0.830),
}
DEVICES = ["小程序", "iOS", "Android", "PC"]
DEVICE_P = [0.37, 0.27, 0.29, 0.07]
DEV_PAY_MULT = {"小程序": 1.03, "iOS": 1.10, "Android": 0.93, "PC": 0.98}
DEV_CART_MULT = {"小程序": 1.02, "iOS": 1.00, "Android": 0.99, "PC": 0.92}

sess_user, sess_date, sess_hour, sess_ch, sess_dev = [], [], [], [], []
user_channel = users["acq_channel"].to_numpy()
intent = np.array([SEG_INTENT[s] for s in seg])

for d_i, d in enumerate(dates):
    n = int(day_counts[d_i])
    eligible = np.nonzero(reg_ts <= d)[0]
    if len(eligible) == 0:
        continue
    p = base_w[eligible]
    p = p / p.sum()
    idx = rng.choice(eligible, n, p=p)
    sess_user.append(idx)
    sess_date.append(np.full(n, d_i))
    sess_hour.append(rng.choice(28, n, p=HOUR_P))

sess_user = np.concatenate(sess_user)
sess_date = np.concatenate(sess_date)
sess_hour = np.concatenate(sess_hour)
n_sess = len(sess_user)

# 渠道：70% 回到自己的获客渠道，30% 随全站分布（自然搜索/信息流仍是主入口）
ch_global = rng.choice(len(CHANNELS), n_sess, p=CHANNEL_P)
ch_own = pd.Categorical(user_channel).codes[sess_user]
use_own = rng.random(n_sess) < 0.70
sess_ch_code = np.where(use_own, ch_own, ch_global)
sess_ch = np.array(CHANNELS)[sess_ch_code]
sess_dev = rng.choice(DEVICES, n_sess, p=DEVICE_P)

p_cart_base = np.array([CH_CONV[c][0] for c in sess_ch])
p_ord_base = np.array([CH_CONV[c][1] for c in sess_ch])
p_pay_base = np.array([CH_CONV[c][2] for c in sess_ch])
dev_pay_mult = np.array([DEV_PAY_MULT[x] for x in sess_dev])
dev_cart_mult = np.array([DEV_CART_MULT[x] for x in sess_dev])
u_intent = intent[sess_user]

p_cart = np.clip(p_cart_base * u_intent * dev_cart_mult, 0.01, 0.92)
p_ord = np.clip(p_ord_base * (0.75 + 0.25 * u_intent), 0.05, 0.90)
p_pay = np.clip(p_pay_base * dev_pay_mult * (0.90 + 0.10 * u_intent), 0.15, 0.97)

# 大促期转化率提升（促销氛围 + 券包刺激）
def promo_conv(d):
    m, dd = d.month, d.day
    if (m, dd) >= (11, 1) and (m, dd) <= (11, 11): return 1.14
    if (m == 5 and dd >= 26) or (m == 6 and dd <= 18): return 1.11
    if m == 12 and 9 <= dd <= 12: return 1.08
    if m == 3 and 5 <= dd <= 8: return 1.07
    if m == 9 and 5 <= dd <= 9: return 1.05
    if (m, dd) >= (1, 28) and (m, dd) <= (2, 4): return 0.92
    return 1.0
conv_f = np.array([promo_conv(d) for d in dates])[sess_date]
p_cart = np.clip(p_cart * conv_f, 0.01, 0.92)
p_pay = np.clip(p_pay * (1 + (conv_f - 1) * 0.35), 0.15, 0.97)

u1, u2, u3 = rng.random(n_sess), rng.random(n_sess), rng.random(n_sess)
cart = u1 < p_cart
order = cart & (u2 < p_ord)
pay = order & (u3 < p_pay)
max_stage = 1 + cart.astype(np.int8) + order.astype(np.int8) + pay.astype(np.int8)

session_id = np.array([f"S{i:07d}" for i in range(1, n_sess + 1)])
sessions = pd.DataFrame({
    "session_id": session_id,
    "user_id": users["user_id"].to_numpy()[sess_user],
    "dt": dates[sess_date].strftime("%Y-%m-%d"),
    "hour": sess_hour,
    "channel": sess_ch,
    "device": sess_dev,
    "max_stage": max_stage,
})
sessions["_date_idx"] = sess_date
sessions["_user_idx"] = sess_user

print(f"[1/4] 会话表：{len(sessions):,} 条，支付会话 {(max_stage==4).sum():,} 条")

# ---------------------------------------------------------------- 4. 订单表
paid = sessions[sessions["max_stage"] == 4].copy()
n_ord = len(paid)

# 用户品类偏好（Dirichlet，制造"偏好稳定"的复购行为）
user_pref = rng.dirichlet(cat_p_adj * 22, N_USERS)
paid_user_idx = paid["_user_idx"].to_numpy()

def promo_label(d):
    m, dd = d.month, d.day
    if (m, dd) >= (11, 1) and (m, dd) <= (11, 11):
        return "双11"
    if (m == 5 and dd >= 26) or (m == 6 and dd <= 18):
        return "618"
    if m == 12 and 9 <= dd <= 12:
        return "双12"
    if m == 3 and 5 <= dd <= 8:
        return "38节"
    if m == 9 and 5 <= dd <= 9:
        return "99节"
    if (m == 1 and dd >= 25) or (m == 2 and dd <= 4):
        return "年货节"
    if m == 12 and dd >= 24:
        return "双旦"
    return "日常"

order_dates = dates[paid["_date_idx"].to_numpy()]
promo = np.array([promo_label(d) for d in order_dates])
# 大促折扣更深
PROMO_DISC = {"日常": (0.03, 0.12), "38节": (0.10, 0.24), "618": (0.14, 0.30),
              "99节": (0.08, 0.20), "双11": (0.18, 0.38), "双12": (0.12, 0.28),
              "年货节": (0.10, 0.22), "双旦": (0.08, 0.20)}
promo_arr = np.array([PROMO_DISC[p] for p in promo])

item_cnt = np.clip(rng.poisson(0.65, n_ord) + 1, 1, 5)

# 品类年度趋势（同比增速），制造品类间的结构性分化
CAT_TREND = {"女装": -0.12, "美妆个护": 0.34, "食品生鲜": 0.16, "3C数码": -0.06,
             "家居日用": 0.07, "母婴玩具": -0.09, "运动户外": 0.41}
day_frac = (order_dates - pd.Timestamp("2025-01-01")).days.to_numpy() / 365.0
trend_mat = np.stack([(1 + CAT_TREND[c]) ** day_frac for c in cat_names], axis=1)  # (n_ord, 7)

recs = []
for i in range(n_ord):
    uid = paid_user_idx[i]
    k = int(item_cnt[i])
    p_cat = user_pref[uid] * trend_mat[i]
    p_cat = p_cat / p_cat.sum()
    cats_pick = rng.choice(len(cat_names), k, p=p_cat)
    for cc in cats_pick:
        c = cat_names[cc]
        pool = prod_by_cat[c]
        w = cat_weight[c]
        pi = rng.choice(pool, p=w)
        qty = 1 if rng.random() < 0.86 else int(rng.integers(2, 4))
        recs.append((i, pi, qty))

items = pd.DataFrame(recs, columns=["oidx", "pidx", "qty"])
items["unit_price"] = prod_price[items["pidx"]]
items["amount"] = items["unit_price"] * items["qty"]
items["category_l1"] = prod_cat[items["pidx"]]
items["cost_rate"] = prod_cost[items["pidx"]]

agg = items.groupby("oidx").agg(
    gmv=("amount", "sum"),
    item_cnt=("qty", "sum"),
    sku_cnt=("pidx", "nunique"),
    main_cat=("category_l1", lambda s: s.value_counts().index[0]),
).reset_index()

disc = rng.uniform(promo_arr[:, 0], promo_arr[:, 1])
# 会员额外折扣
level_mult = pd.Categorical(users["member_level"].to_numpy()[paid_user_idx]).codes
disc = np.clip(disc + np.array([0.03, 0.02, 0.01])[np.minimum(level_mult, 2)], 0, 0.55)

main_refund = np.array([CATS[c]["refund"] for c in agg["main_cat"]])
ch_refund_mult = np.array([{"直播引流": 1.35, "信息流广告": 1.12, "社交裂变": 1.05,
                            "自然搜索": 0.92, "私域社群": 0.86}[c] for c in paid["channel"]])
refund_p = np.clip(main_refund * ch_refund_mult, 0.005, 0.35)
refund_flag = rng.random(n_ord) < refund_p

gmv = agg["gmv"].to_numpy()
pay_amount = np.round(gmv * (1 - disc), 2)
refund_amount = np.where(refund_flag, pay_amount * np.where(rng.random(n_ord) < 0.3, 0.5, 1.0), 0.0)

orders = pd.DataFrame({
    "order_id": [f"O{i:07d}" for i in range(1, n_ord + 1)],
    "session_id": paid["session_id"].to_numpy(),
    "user_id": paid["user_id"].to_numpy(),
    "order_date": order_dates.strftime("%Y-%m-%d"),
    "order_hour": paid["hour"].to_numpy(),
    "channel": paid["channel"].to_numpy(),
    "device": paid["device"].to_numpy(),
    "promo": promo,
    "main_category": agg["main_cat"].to_numpy(),
    "sku_cnt": agg["sku_cnt"].to_numpy(),
    "item_cnt": agg["item_cnt"].to_numpy(),
    "gmv": np.round(gmv, 2),
    "discount_rate": np.round(disc, 4),
    "pay_amount": pay_amount,
    "refund_flag": refund_flag.astype(int),
    "refund_amount": np.round(refund_amount, 2),
})
orders = orders.sort_values(["order_date", "order_hour"]).reset_index(drop=True)
first_order = ~orders.duplicated("user_id", keep="first")
reg_map = pd.to_datetime(users.set_index("user_id")["reg_date"])
reg_gap = (pd.to_datetime(orders["order_date"]) - orders["user_id"].map(reg_map)).dt.days
# 业务口径：注册 90 天内完成首购 = 新客；注册已久才下单 = 老客回流
orders["is_new_buyer"] = (first_order & (reg_gap <= 90)).astype(int)

items_out = items.copy()
items_out["order_id"] = orders["order_id"].to_numpy()[items_out["oidx"].to_numpy()]
items_out["product_id"] = products["product_id"].to_numpy()[items_out["pidx"].to_numpy()]
items_out = items_out[["order_id", "product_id", "category_l1", "qty", "unit_price", "amount", "cost_rate"]]
items_out = items_out.rename(columns={"qty": "quantity"})
items_out.insert(2, "category_l2", products["category_l2"].to_numpy()[items["pidx"].to_numpy()])

# ---------------------------------------------------------------- 5. 落库
sessions_out = sessions.drop(columns=["_date_idx", "_user_idx"])
sessions_out.to_csv(os.path.join(RAW, "sessions.csv"), index=False, encoding="utf-8-sig")
orders.to_csv(os.path.join(RAW, "orders.csv"), index=False, encoding="utf-8-sig")
items_out.to_csv(os.path.join(RAW, "order_items.csv"), index=False, encoding="utf-8-sig")
users.to_csv(os.path.join(RAW, "users.csv"), index=False, encoding="utf-8-sig")
products.to_csv(os.path.join(RAW, "products.csv"), index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 6. 体检
st = sessions_out["max_stage"]
n = len(sessions_out)
print(f"[2/4] 漏斗：浏览 {n:,} -> 加购 {(st>=2).sum():,}({(st>=2).mean():.2%}) "
      f"-> 下单 {(st>=3).sum():,}({(st>=3).mean():.2%}) -> 支付 {(st>=4).sum():,}({(st>=4).mean():.2%})")
net = orders["pay_amount"].sum() - orders["refund_amount"].sum()
print(f"[3/4] 订单 {len(orders):,} 笔 | 买家 {orders.user_id.nunique():,} 人 | "
      f"GMV ¥{orders.gmv.sum():,.0f} | 实付 ¥{orders.pay_amount.sum():,.0f} | 净G... ¥{net:,.0f}")
print(f"[4/4] 客单价 ¥{orders.pay_amount.mean():.1f} | 退款率 {orders.refund_flag.mean():.2%} | "
      f"复购率 {(orders.groupby('user_id').size() > 1).mean():.2%}")
print("文件已写入 data/raw/")
