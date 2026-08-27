# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 21:24:23 2026

@author: WeiCH
"""

# -*- coding: utf-8 -*-
"""
Figure 2 (Horizontal Layout - 1 Row x 2 Columns): 
Spatial patterns of R95pTOT change (P4 - P1) sharing a single Colorbar.
"""

import glob, os, warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
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

def agreement_mask(da, dim='member', min_agree=26):
    pos = (da > 0).sum(dim)
    neg = (da < 0).sum(dim)
    agree = np.maximum(pos, neg)
    return agree >= min_agree

# =============================================================================
# 3. 数据处理逻辑
# =============================================================================
def process_map_data(pr_path, p1_span, p4_span):
    pr_files = sorted(glob.glob(pr_path))
    main_map_list = []

    for pf in pr_files:
        ds_p = xr.open_dataset(pf)
        pr = ds_p['prcptot'].where((ds_p.lat >= -60) & (ds_p.lat <= 90), drop=True)
        
        baseline = pr.sel(year=slice(1995, 2014)).mean('year')
        p1 = pr.sel(year=slice(p1_span[0], p1_span[1])).mean('year')
        p4 = pr.sel(year=slice(p4_span[0], p4_span[1])).mean('year')
        
        diff_pct = (p4 - p1) / baseline * 100
        main_map_list.append(diff_pct)

    le_map = xr.concat(main_map_list, dim='member')
    lem_map = le_map.mean('member')#.where(mask > 0)
    agree_map = agreement_mask(le_map)#.where(mask > 0)

    return {
        'map_data': lem_map,
        'map_agree': agree_map
    }

# 读取数据
data_126 = process_map_data(
    pr_path=r'E:\data\etccdi\ssp126\prcptot\*.nc',
    p1_span=(2170, 2190), p4_span=(2280, 2300)
)

data_534 = process_map_data(
    pr_path=r'E:\data\etccdi\ssp534\prcptot\*.nc',
    p1_span=(2165, 2185), p4_span=(2280, 2300)
)

# =============================================================================
# 4. 画布与 GridSpec 左右单行布局 (1行2列)
# =============================================================================
fig = plt.figure(figsize=(8, 2.5))

gs = gridspec.GridSpec(
    1, 2,
    figure=fig,
    left=0.08, right=0.95,
    bottom=0.22, top=0.88,
    wspace=0.25
)

map_ticks = [-30, -20, -10, -8, -5, -3, -1, 0, 1, 3, 5, 8, 10, 20, 30]
norm_map = BoundaryNorm(boundaries=map_ticks, ncolors=plt.get_cmap('BrBG').N)

# =============================================================================
# 5. 地图绘制函数
# =============================================================================
def draw_map_subplot(ax, dataarray, sig_mask, panel_label):
    ax.set_aspect('auto')
    
    im = dataarray.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap='BrBG', norm=norm_map,
        add_labels=False, add_colorbar=False
    )

    mpl.rcParams['hatch.linewidth'] = 0.6
    sig_cyc, lon_cyc = add_cyclic_point(sig_mask.values, coord=sig_mask.lon.values)
    ax.contourf(lon_cyc, sig_mask.lat, sig_cyc,
                levels=[0.5, 1.5], colors='none', hatches=['.....'],
                transform=ccrs.PlateCarree())
    
    # Amazon 标注框
    rect_amazon = mpatches.Rectangle(
        xy=(-60, -10), width=120, height=30,
        facecolor='none', edgecolor='red', linewidth=1.0, linestyle='--',
        transform=ccrs.PlateCarree(), zorder=6
    )
    ax.add_patch(rect_amazon)
    
    # Eurasia 标注框
    poly_vertices = [(-10, 65), (145, 65), (145, 20), (105, 20), (105, 50), (-10, 50)]
    poly_eurasia = mpatches.Polygon(
        poly_vertices, closed=True,
        facecolor='none', edgecolor='red', linewidth=1.0, linestyle='--',
        transform=ccrs.PlateCarree(), zorder=6
    )
   # ax.add_patch(poly_eurasia)

    ax.coastlines(linewidth=0.5)
    ax.set_xticks(np.arange(-180, 181, 120), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-60, 91, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis='both', labelsize=8.5, length=2, width=0.5)
    
    ax.set_title(f"{panel_label} PRCPTOT", fontsize=10, fontweight='bold', loc='left')
    ax.set_title(f"P4-P3", fontsize=10, loc='right')
    return im

# =============================================================================
# 6. 组装子图并绘制
# =============================================================================
ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax_a.text(
    0.5,
    1.2,
    'SSP1-2.6',
    transform=ax_a.transAxes,
    fontsize=10,
    fontweight='bold',
    ha='center',
    va='bottom',
    )
ax_b.text(
    0.5,
    1.2,
    'SSP5-3.4-OS',
    transform=ax_b.transAxes,
    fontsize=10,
    fontweight='bold',
    ha='center',
    va='bottom',
    )


im_a = draw_map_subplot(ax_a, data_126['map_data'], data_126['map_agree'], '(a)')
im_b = draw_map_subplot(ax_b, data_534['map_data'], data_534['map_agree'], '(b)')

# =============================================================================
# 7. 共享 Colorbar 自动跨越 ax_a 与 ax_b 居中放置
# =============================================================================
fig.canvas.draw()

pos_a = ax_a.get_position()
pos_b = ax_b.get_position()

# 计算横跨 ax_a 到 ax_b 的总宽度，并按 0.7 比例居中
row_left, row_right = pos_a.x0, pos_b.x1
row_width = row_right - row_left
cb_width = row_width * 0.7
cb_left = row_left + (row_width - cb_width) / 2
cb_bottom = pos_a.y0 - 0.15
cb_height = 0.03

cax = fig.add_axes([cb_left, cb_bottom, cb_width, cb_height])
clean_map_ticks = [-60, -20, -8, -3, 0, 3, 8, 20, 60]

cb = fig.colorbar(im_a, cax=cax, orientation='horizontal', ticks=clean_map_ticks)
cb.ax.tick_params(labelsize=8.5, rotation=0)

cax.text(
    1.02, 0.5, 
    r'%', 
    transform=cax.transAxes, 
    fontsize=9.5, 
    va='center'
)

# 保存与显示
plt.savefig('FigureS4.pdf', dpi=800, bbox_inches='tight')
plt.savefig('FigureS4.png', dpi=800, bbox_inches='tight')

plt.show()