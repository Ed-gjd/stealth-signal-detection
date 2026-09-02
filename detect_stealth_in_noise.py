# -*- coding: utf-8 -*-
"""
隐身目标（小σ）在噪声里还能不能检测到？——直接算给数据看
流程：雷达方程算SNR → 理论检测概率(Marcum Q) → 蒙特卡洛实测验证 → 距离剖面图
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import i0e
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# ============ 1. 场景参数（X波段多功能雷达） ============
c   = 3e8
lam = 0.03          # X 波段 3cm
Pt  = 1e3           # 峰值功率 1 kW
G   = 10**(35/10)   # 天线增益 35 dB
B   = 1e6           # 带宽 1 MHz
kT0 = 4.0e-21       # k·T0 @ 290K
F   = 3.0           # 噪声系数(线性)
L   = 10**(3/10)    # 系统损耗 3 dB
Np  = 16            # 相干积累脉冲数
Pfa = 1e-6          # 虚警率

def snr_single_db(R_km, sigma):
    """单脉冲 SNR（雷达方程，匹配滤波后）"""
    R = R_km*1e3
    Pr = Pt*G*G*lam**2*sigma/((4*np.pi)**3*R**4)
    N = kT0*B*F*L
    return 10*np.log10(Pr/N)

def pd_marcum(snr_db):
    """单脉冲、平方律检波、非起伏目标，P_d = Marcum Q1(a,b)
       Q1(a,b) = ∫_b^∞ x exp(-(x²+a²)/2) I0(ax) dx
       用指数缩放贝塞尔 I0e 做数值稳定：∫ x exp(-(x-a)²/2) i0e(ax) dx
    """
    snr = 10**(snr_db/10)
    a = np.sqrt(2*snr)
    b = np.sqrt(-2*np.log(Pfa))
    # 积分上限用 max(a,b)+10：高斯尾部 10σ 外贡献 <1e-14，且避免 quad 在
    # 大 a 时（峰窄）从 b 到 ∞ 的初始采样完全错过峰导致误判为 0
    f = lambda x: x*np.exp(-(x-a)**2/2)*i0e(a*x)
    val, _ = quad(f, b, max(a, b) + 10)
    return min(max(val, 0.0), 1.0)

# ============ 2. 数据表：σ × 距离 → SNR 与 P_d ============
sigmas = [1.0, 0.1, 0.01, 0.001]
ranges = [5, 10, 20, 40]
integ_gain = 10*np.log10(Np)   # +12 dB

print('='*72)
print(f'SNR(dB) = 单脉冲SNR + {Np}脉冲相干积累(+{integ_gain:.0f}dB)')
print('σ单位 m²；普通飞机~几~几十，隐身目标可到 0.01~0.001 量级')
print('距离(km) |   σ=1.0     0.1     0.01    0.001')
for R in ranges:
    row = [snr_single_db(R,s)+integ_gain for s in sigmas]
    print(f'  {R:5d}   |' + ''.join(f'{v:8.1f}' for v in row))

print('-'*72)
print('理论检测概率 P_d（P_fa=1e-6，单脉冲非起伏）  <50%基本检不到 >90%可靠')
print('距离(km) |   σ=1.0     0.1     0.01    0.001')
for R in ranges:
    row = [pd_marcum(snr_single_db(R,s)+integ_gain) for s in sigmas]
    print(f'  {R:5d}   |' + ''.join(f'{v:8.0%}' for v in row))
print('='*72)

# 找一下 P_d=0.5/0.9/0.99 需要多少 SNR（经典锚点）
print('SNR → P_d 扫描（P_fa=1e-6，单脉冲非起伏）：')
for db in [6, 8, 10, 12, 14, 16, 18]:
    print(f'  SNR={db:4.1f}dB → P_d = {pd_marcum(db):6.1%}')
for target in [0.5, 0.9, 0.99]:
    best = min((db/10 for db in range(0, 200)),
               key=lambda x: abs(pd_marcum(x)-target))
    print(f'  P_d={target:.2f} 需要 SNR≈{best:.1f}dB')

# ============ 3. 蒙特卡洛验证：实测 P_d vs 理论 ============
def mc_pd(snr_db, trials=20000):
    snr = 10**(snr_db/10)
    A = np.sqrt(snr)
    th = np.sqrt(-np.log(Pfa))     # 幅度门限 = sqrt(功率门限 -ln(Pfa))
    hits = 0
    for _ in range(trials):
        z = A + (np.random.randn()+1j*np.random.randn())/np.sqrt(2)
        if np.abs(z) > th:
            hits += 1
    return hits/trials

print('\n蒙特卡洛实测 vs 理论（各2万次实验）：')
for db in [8, 12, 16]:
    print(f'  SNR={db:3d}dB  实测 P_d={mc_pd(db):.3f}  理论 P_d={pd_marcum(db):.3f}')

# ============ 4. 距离剖面：埋在噪声里 vs 明显可检 ============
fig, ax = plt.subplots(2, 1, figsize=(10, 6.5))
for a, db, title in [(ax[0], 10, 'SNR=10dB：目标埋在噪声里（只有噪声时它长这样）'),
                     (ax[1], 20, 'SNR=20dB：目标峰明显高出门限')]:
    A = np.sqrt(10**(db/10))
    n = 256
    z = (np.random.randn(n)+1j*np.random.randn(n))/np.sqrt(2)
    z[128] += A
    a.plot(np.abs(z)**2, lw=0.8)
    a.axhline(-2*np.log(Pfa), color='r', ls='--', label='检测门限 (P_fa=1e-6)')
    a.axvline(128, color='g', ls=':', label='目标真实位置')
    a.set_title(title); a.set_ylabel('|z|² (功率)')
    a.legend(loc='upper right')
ax[1].set_xlabel('距离单元')
plt.tight_layout()
plt.savefig('detect_in_noise.png', dpi=120)
print('\n图已保存: detect_in_noise.png')
