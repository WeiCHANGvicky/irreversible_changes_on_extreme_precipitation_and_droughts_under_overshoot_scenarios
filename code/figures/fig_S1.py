# -*- coding: utf-8 -*-
"""
Figure 2: Spatial patterns and regional climate trajectory with inset KS-test boxplots.
Hatching non-significant areas with customizable hatch styles and corrected inset positions.
"""

import glob, os, warnings
import numpy as np
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

warnings.filterwarnings('ignore')

# =============================================================================
# 1. 全局出版级样式配置
# =============================================================================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']

# =============================================================================
# 2. 基础数据与掩码准备
# =============================================================================
mask_file = r"E:\fx\sftlf\sftlf_fx_ACCESS-ESM1-5.nc"
area_file = r"E:\fx\areacella\areacella_fx_ACCESS-ESM1-5.nc"

mask = xr.open_dataset(mask_file)["sftlf"] / 100 
area = xr.open_dataset(area_file)['areacella'] 

def area_weighted_mean(data, weights):
    num = (data * weights).sum(dim=("lat", "lon"), skipna=True)
    den = weights.sum(dim=("lat", "lon"), skipna=True)
    return num / den

def agreement_mask(da, dim='member', min_agree=26):
    pos = (da > 0).sum(dim)
    neg = (da < 0).sum(dim)
    agree = np.maximum(pos, neg)
    return agree >= min_agree

# =============================================================================
# 3. 区域掩码提取函数（严格对应地图上的画框范围）
# =============================================================================
def extract_region_data(ds, region_name):
    if region_name == 'Amazon':
        # Amazon 矩形框: Lat -10~15, Lon 295~328 (-65~-32)
        cond = (ds.lat >= -10) & (ds.lat <= 15) & (ds.lon >= 295) & (ds.lon <= 328)
    elif region_name == 'Mid-latitude Eurasia':
        # 不规则多边形 Eurasia: [(-10, 65), (145, 65), (145, 20), (105, 20), (105, 50), (-10, 50)]
        # Lon 在 0~360 坐标系下，-10度对应 350度
        # 区域1: 经度 -10°~105° (即 >=350 或 <=105), 纬度 50°~65°
        cond1 = (ds.lat >= 50) & (ds.lat <= 65) & ((ds.lon >= 350) | (ds.lon <= 105))
        # 区域2: 经度 105°~145°, 纬度 20°~65°
        cond2 = (ds.lat >= 20) & (ds.lat <= 65) & (ds.lon > 105) & (ds.lon <= 145)
        
        cond = cond1 | cond2
    
    ds_reg = ds.where(cond, drop=True)
    area_reg = area.where(cond, drop=True)
    mask_reg = mask.where(cond, drop=True)
    
    weights = area_reg * mask_reg
    return area_weighted_mean(ds_reg, weights)

# =============================================================================
# 4. 数据处理逻辑
# =============================================================================
def process_scenario_data(pr_path, tas_path, p1_span, p4_span, target_temp):
    pr_files = sorted(glob.glob(pr_path))
    tas_files = sorted(glob.glob(tas_path))

    # 1. 计算 GMST
    gmst_list = []
    for tf in tas_files:
        ds_t = xr.open_dataset(tf)
        tas = area_weighted_mean(ds_t['tas'] - 273.15, area)
        gmst = tas - tas.sel(time=slice('1850','1900')).mean('time')
        gmst_list.append(gmst)
    le_gmst = xr.concat(gmst_list, dim='member')
    lem_gmst = le_gmst.mean('member').assign_coords(year=gmst_list[0]['time'].dt.year).swap_dims({'time':'year'}).sel(year=slice(1980,2300))
    smoothed_gmst = lem_gmst.rolling(year=21, center=True).mean('year')

    # 2. 计算 R95pTOT
    main_map_list = []
    eurasia_p1_members, eurasia_p4_members = [], []
    amazon_p1_members, amazon_p4_members = [], []
    
    pr_series_eurasia_list, pr_series_amazon_list = [], []

    for pf in pr_files:
        ds_p = xr.open_dataset(pf)
        # 安全选取纬度范围，避免因为正序/倒序导致的 slice 提取为空问题
        pr = ds_p['r95pdays'].where((ds_p.lat >= -60) & (ds_p.lat <= 90), drop=True)
        
        baseline = pr.sel(year=slice(1995, 2014)).mean('year')
        p1 = pr.sel(year=slice(p1_span[0], p1_span[1])).mean('year')
        p4 = pr.sel(year=slice(p4_span[0], p4_span[1])).mean('year')
        
        # 2.1 空间地图：相对变化率 (%)
        diff_pct = (p4 - p1) / baseline * 100
        main_map_list.append(diff_pct)
        
        # 2.2 计算 P1 和 P4 相对于 1995-2014 基线期的距平百分比 (%) 用于插图箱线图
        p1_anom = (p1 - baseline) / baseline * 100
        p4_anom = (p4 - baseline) / baseline * 100
        
        eurasia_p1_members.append(float(extract_region_data(p1_anom, 'Mid-latitude Eurasia')))
        eurasia_p4_members.append(float(extract_region_data(p4_anom, 'Mid-latitude Eurasia')))
        amazon_p1_members.append(float(extract_region_data(p1_anom, 'Amazon')))
        amazon_p4_members.append(float(extract_region_data(p4_anom, 'Amazon')))
        
        # 2.3 计算逐年时间序列相对于 1995-2014 基线期的距平百分比 (%)
        ts_eu_raw = extract_region_data(pr, 'Mid-latitude Eurasia')
        ts_am_raw = extract_region_data(pr, 'Amazon')

        base_eu = ts_eu_raw.sel(year=slice(1995, 2014)).mean('year')
        base_am = ts_am_raw.sel(year=slice(1995, 2014)).mean('year')

        ts_eu_anom = (ts_eu_raw - base_eu) / base_eu * 100
        ts_am_anom = (ts_am_raw - base_am) / base_am * 100

        pr_series_eurasia_list.append(ts_eu_anom)
        pr_series_amazon_list.append(ts_am_anom)

    le_map = xr.concat(main_map_list, dim='member')
    lem_map = le_map.mean('member').where(mask > 0)
    agree_map = agreement_mask(le_map).where(mask > 0)

    lem_eu = xr.concat(pr_series_eurasia_list, dim='member').mean('member').sel(year=slice(1980,2300))
    smoothed_eu = lem_eu.rolling(year=21, center=True).mean('year')
    
    lem_am = xr.concat(pr_series_amazon_list, dim='member').mean('member').sel(year=slice(1980,2300))
    smoothed_am = lem_am.rolling(year=21, center=True).mean('year')

    return {
        'map_data': lem_map,
        'map_agree': agree_map,
        'gmst_smooth': smoothed_gmst,
        'eu_smooth': smoothed_eu,
        'am_smooth': smoothed_am,
        'eu_ks': (np.array(eurasia_p1_members), np.array(eurasia_p4_members)),
        'am_ks': (np.array(amazon_p1_members), np.array(amazon_p4_members))
    }

# 读取数据
data_126 = process_scenario_data(
    pr_path=r'E:\data\etccdi\ssp126\r95pdays\*.nc',
    tas_path=r'E:\data\tas\annual\ssp126\*.nc',
    p1_span=(2035, 2055), p4_span=(2280, 2300), target_temp=1.83
)

data_534 = process_scenario_data(
    pr_path=r'E:\data\etccdi\ssp534\r95pdays\*.nc',
    tas_path=r'E:\data\tas\annual\ssp534\*.nc',
    p1_span=(2025, 2045), p4_span=(2280, 2300), target_temp=1.866
)

# =============================================================================
# 5. 画布与 GridSpec 矩阵对齐布局
# =============================================================================
fig = plt.figure(figsize=(8, 5.0))

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    left=0.08, right=0.96,
    bottom=0.18, top=0.93,
    wspace=0.32, hspace=0.45,
    width_ratios=[1.5, 1.2, 1.2]
)

map_ticks = [-100, -60, -30, -20, -10, -8, -5, -3, -1, 0, 1, 3, 5, 8, 10, 20, 30, 60, 100]
norm_map = BoundaryNorm(boundaries=map_ticks, ncolors=plt.get_cmap('BrBG').N)

# =============================================================================
# 6. 子图绘制函数
# =============================================================================
def draw_map_subplot(ax, dataarray, sig_mask, panel_label,var_label, scenario_title):
    ax.set_aspect('auto')
    
    im = dataarray.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap='BrBG', norm=norm_map,
        add_labels=False, add_colorbar=False
    )
    import matplotlib as mpl


    mpl.rcParams['hatch.linewidth'] = 0.6
    # 绘制不显著区域的斜线
    sig_cyc, lon_cyc = add_cyclic_point(sig_mask.values, coord=sig_mask.lon.values)
    ax.contourf(lon_cyc, sig_mask.lat, sig_cyc,
                levels=[0.5, 1.5], colors='none', hatches=['.....'],
                transform=ccrs.PlateCarree())
    
    # Amazon 标注框
    rect_amazon = mpatches.Rectangle(
        xy=(-65, -10), width=33, height=25,
        facecolor='none', edgecolor='red', linewidth=1.0, linestyle='--',
        transform=ccrs.PlateCarree(), zorder=6
    )
    ax.add_patch(rect_amazon)
    
    # Eurasia 标注框（不规则多边形）
    poly_vertices = [(-10, 65), (145, 65), (145, 20), (105, 20), (105, 50), (-10, 50)]
    poly_eurasia = mpatches.Polygon(
        poly_vertices, closed=True,
        facecolor='none', edgecolor='red', linewidth=1.0, linestyle='--',
        transform=ccrs.PlateCarree(), zorder=6
    )
    ax.add_patch(poly_eurasia)

    ax.coastlines(linewidth=0.5)
    ax.set_xticks(np.arange(-180, 181, 120), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-60, 91, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_ylabel(f' {scenario_title}', fontsize=10, fontweight='bold')
    ax.tick_params(axis='both', labelsize=9, length=2, width=0.5)
    
    ax.set_title(f"{panel_label} {var_label}", fontsize=10, fontweight='bold', loc='left')
    ax.set_title("P4-P1", fontsize=10, loc='right')
    return im

def draw_scatter_subplot(ax, gmst_ts, pr_ts, target_temp, panel_label, region_title, ks_p1, ks_p4, inset_pos=[0.58, 0.12, 0.35, 0.32], target_years=None):
    common_years = np.intersect1d(gmst_ts['year'], pr_ts['year'])
    x_vals = gmst_ts.sel(year=common_years).values
    y_vals = pr_ts.sel(year=common_years).values

    sc = ax.scatter(x_vals, y_vals, c=common_years, cmap='turbo', s=12, edgecolor='none', zorder=3, vmin=1980, vmax=2300)
    ax.axvline(target_temp, linestyle=':', color='gray', linewidth=1, zorder=1)

    default_labels = ['P1', 'P2', 'P3', 'P4']

    # 处理自定义年份、文字及偏移量
    if target_years is not None:
        for i, item in enumerate(target_years):
            # 结构解析: (年份, 标注文字, (x_offset, y_offset))
            if isinstance(item, (list, tuple)):
                if len(item) == 3:
                    yr, label, xytext = item
                elif len(item) == 2:
                    yr, xytext = item
                    label = default_labels[i] if i < len(default_labels) else f'P{i+1}'
                else:
                    yr = item[0]
                    label = default_labels[i] if i < len(default_labels) else f'P{i+1}'
                    xytext = (3, 3)
            else:
                yr = item
                label = default_labels[i] if i < len(default_labels) else f'P{i+1}'
                xytext = (3, 3)

            if yr in common_years:
                idx = np.where(common_years == yr)[0][0]
                x_pos = x_vals[idx]
                y_pos = y_vals[idx]
                
                # 1. 绘制黑圈
                ax.scatter(x_pos, y_pos, s=25, marker='o', facecolors='none', edgecolors='black', linewidths=1.0, zorder=4)
                
                # 2. 绘制文本（使用独立设置的偏移量）
                ax.annotate(
                    label, 
                    (x_pos, y_pos),
                    xytext=xytext, textcoords='offset points',
                    fontsize=9, fontweight='bold', color='black',
                    zorder=5
                )

    ax.set_xlabel('GMST (K)', fontsize=10, labelpad=2)
    ax.set_ylabel('R95pFreq Anomaly (%)', fontsize=10, labelpad=2)
    ax.set_title(f'{panel_label}', fontsize=10, fontweight='bold', loc='left')
    ax.set_title(f'{region_title}', fontsize=10, loc='right')
    
    ax.tick_params(axis='both', labelsize=9, length=2, width=0.5)

    # 嵌入箱线图
    ax_inset = ax.inset_axes(inset_pos)
    ax_inset.set_facecolor('none')  # <<-- 设置子图背景为透明
    arr_p1 = np.asarray(ks_p1, dtype=float).flatten()
    arr_p4 = np.asarray(ks_p4, dtype=float).flatten()

    ks_stat, p_val = stats.ks_2samp(arr_p1, arr_p4)
    p_val_scalar = float(p_val)
    p_text = r'$p$ < 0.01' if p_val_scalar < 0.01 else f'$p$ = {p_val_scalar:.2f}'

    bp = ax_inset.boxplot(
        [arr_p1, arr_p4],
        whis=[5, 95],
        patch_artist=True,
        showfliers=False,
        widths=0.5,
        medianprops=dict(color='black', linewidth=0.8),
        flierprops=dict(markersize=2)
    )

    colors = ['#004e85', '#8f1f22']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax_inset.set_xticklabels(['P1', 'P4'], fontsize=8)
    ax_inset.tick_params(axis='y', labelsize=7, pad=0.8)
    ax_inset.set_title(p_text, fontsize=9, pad=2)
    ax_inset.grid(False)

    return sc

# =============================================================================
# 7. 组装子图并绘制
# =============================================================================
ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax_c = fig.add_subplot(gs[0, 1])
ax_e = fig.add_subplot(gs[0, 2])

ax_b = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax_d = fig.add_subplot(gs[1, 1])
ax_f = fig.add_subplot(gs[1, 2])

im_a = draw_map_subplot(ax_a, data_126['map_data'], data_126['map_agree'], '(a)', 'R95pFreq','SSP1-2.6')
im_b = draw_map_subplot(ax_b, data_534['map_data'], data_534['map_agree'], '(b)', 'R95pFreq', 'SSP5-3.4-OS')

# =============================================================================
# 在这里独立配置每个子图中 P1-P4 的 (年份, 标签文字, (x像素偏移, y像素偏移))
# 说明：(3, 3) 代表向右上偏离；(-12, -8) 代表向左下偏离；(4, -10) 代表向右下偏离
# =============================================================================
targets_c = [
    (2045, 'P1', (-6, -12)),
    (2075, 'P2', (-6, -18)),
    (2180, 'P3', (-15, 1)),
    (2290, 'P4', (-15, -6))
]

targets_e = [
    (2045, 'P1', (-15, -3)),
    (2075, 'P2', (-20, -5)),
    (2180, 'P3', (3, 1)),
    (2290, 'P4', (3, 3))
]

targets_d = [
    (2035, 'P1', (5, -10)),
    (2060, 'P2', (-5, -15)),
    (2150, 'P3', (-18, 1)),
    (2290, 'P4', (-15, -8))
]

targets_f = [
    (2035, 'P1', (-12, -12)),
    (2060, 'P2', (-7, 8)),
    (2150, 'P3', (4, -3)),
    (2290, 'P4', (6, -4))
]

# SSP1-2.6 子图 (c) 与 (e)
sc_c = draw_scatter_subplot(
    ax_c, data_126['gmst_smooth'], data_126['eu_smooth'], 1.83, '(c)', 'Mid-latitude Eurasia', 
    data_126['eu_ks'][0], data_126['eu_ks'][1], 
    inset_pos=[0.15, 0.56, 0.35, 0.32], target_years=targets_c
)

sc_e = draw_scatter_subplot(
    ax_e, data_126['gmst_smooth'], data_126['am_smooth'], 1.83, '(e)', 'Amazon', 
    data_126['am_ks'][0], data_126['am_ks'][1], 
    inset_pos=[0.21, 0.18, 0.35, 0.32], target_years=targets_e
)

# SSP5-3.4 子图 (d) 与 (f)
sc_d = draw_scatter_subplot(
    ax_d, data_534['gmst_smooth'], data_534['eu_smooth'], 1.866, '(d)', 'Mid-latitude Eurasia', 
    data_534['eu_ks'][0], data_534['eu_ks'][1], 
    inset_pos=[0.13, 0.56, 0.35, 0.32], target_years=targets_d
)

sc_f = draw_scatter_subplot(
    ax_f, data_534['gmst_smooth'], data_534['am_smooth'], 1.866, '(f)', 'Amazon', 
    data_534['am_ks'][0], data_534['am_ks'][1], 
    inset_pos=[0.138, 0.14, 0.35, 0.32], target_years=targets_f
)

# =============================================================================
# 8. 动态 Colorbar 对齐
# =============================================================================
fig.canvas.draw()

pos_b = ax_b.get_position()
pos_d = ax_d.get_position()
pos_f = ax_f.get_position()


cax_map = fig.add_axes([pos_b.x0, 0.08, pos_b.width, 0.018])
clean_map_ticks = [-60, -20, -8,-3, 0, 3,8,20, 60]
cb_map = fig.colorbar(im_a, cax=cax_map, orientation='horizontal', ticks=clean_map_ticks)
cb_map.ax.tick_params(labelsize=9, rotation=0)

cax_map.text(
    1.03, 0.5, 
    r'%', 
    transform=cax_map.transAxes, 
    fontsize=10, 
    va='center',      # 垂直居中，确保处于 colorbar 正右侧
 #   ha='left'        # 文本向右延伸
     )

row2_left, row2_right = pos_d.x0, pos_f.x1
row2_width = row2_right - row2_left
cb_width = row2_width * 0.8
cb_left = row2_left + (row2_width - cb_width) / 2


scatter_width = pos_f.x1 - pos_d.x0
cax_year = fig.add_axes([cb_left, 0.08, cb_width, 0.018])
cb_year = fig.colorbar(sc_c, cax=cax_year, orientation='horizontal')
cb_year.ax.tick_params(labelsize=9,rotation=0)
cax_year.text(
    1.03, 0.5, 
    r'Year', 
    transform=cax_year.transAxes, 
    fontsize=10, 
    va='center',      # 垂直居中，确保处于 colorbar 正右侧
 #   ha='left'        # 文本向右延伸
)


plt.savefig('FigureS1.pdf', dpi=800, bbox_inches='tight')
plt.savefig('FigureS1.png', dpi=800, bbox_inches='tight')

plt.show()