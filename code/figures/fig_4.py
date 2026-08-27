# -*- coding: utf-8 -*-
"""
Figure 4: Mid-latitude Eurasian Baroclinicity, Water Vapor, and Extreme Precipitation
Journal: Geophysical Research Letters (GRL)
Author: WeiCH
"""

import os
import glob
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

warnings.filterwarnings('ignore')

# =============================================================================
# 0. 全局出版级样式设置 (GRL 标准)
# =============================================================================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['hatch.linewidth'] = 0.4

# =============================================================================
# 1. 数据处理函数
# =============================================================================
def agreement_mask(da, dim='member', min_agree=26):
    """计算跨模式符号一致性 Mask"""
    pos = (da > 0).sum(dim)
    neg = (da < 0).sum(dim)
    agree = xr.ufuncs.maximum(pos, neg) if hasattr(xr, 'ufuncs') else np.maximum(pos, neg)
    return agree >= min_agree

def calc_tas_hyster(file_pattern, slice_p2, slice_p4, min_agree=26):
    """计算 Tas (P4-P2) 集合平均与 Mask"""
    files = sorted(glob.glob(file_pattern))
    hyster_list = []
    for file in files:
        ds = xr.open_dataset(file)
        tas = ds['tas'] - 273.15
        
        p2 = tas.sel(time=slice(str(slice_p2[0]), str(slice_p2[1]))).mean('time')
        p4 = tas.sel(time=slice(str(slice_p4[0]), str(slice_p4[1]))).mean('time')
        hyster_list.append(p4 - p2)
        
    le_hyster = xr.concat(hyster_list, dim='member')
    lem_hyster = le_hyster.mean('member')
    agree_hyster = agreement_mask(le_hyster, min_agree=min_agree)
    return lem_hyster, agree_hyster

def calc_wvc_cooling(ssp_folder_pattern, slice_p2, slice_p4, min_agree=30):
    """计算 WVC (P4-P2) 集合平均与 Mask"""
    cooling_list = []
    folders = sorted(glob.glob(ssp_folder_pattern))
    for folder in folders:
        folder_name = os.path.basename(folder)
        hist = xr.open_dataset(fr'E:\data\prw\historical_year\prw_Amon_ACCESS-ESM1-5_historical_{folder_name}1_gn_185001-201412.nc')
        prw_hist = hist['prw']
        
        files = sorted(glob.glob(os.path.join(folder, '*.nc')))
        prw_ssp = xr.open_mfdataset(files, combine="by_coords")['prw']
        
        p2 = prw_ssp.sel(year=slice(slice_p2[0], slice_p2[1])).mean('year')
        p4 = prw_ssp.sel(year=slice(slice_p4[0], slice_p4[1])).mean('year')
        
        cooling = (p4 - p2)
        cooling_list.append(cooling)
        
    le_cooling = xr.concat(cooling_list, dim='member')
    lem_cooling = le_cooling.mean('member')
    agree_cooling = agreement_mask(le_cooling, min_agree=min_agree)
    return lem_cooling, agree_cooling

def calc_omega_histogram(wap_pattern, sftlf_file, areacella_file, slice_p2, slice_p4):
    """计算欧亚大陆陆地 Omega500 数据统计"""
    sftlf = xr.open_dataset(sftlf_file)["sftlf"] / 100.0
    areacella = xr.open_dataset(areacella_file)["areacella"]

    sftlf = sftlf.assign_coords(lon=((sftlf["lon"] + 180) % 360 - 180)).sortby("lon")
    areacella = areacella.assign_coords(lon=((areacella["lon"] + 180) % 360 - 180)).sortby("lon")
    land_area_grid = areacella * sftlf

    files = sorted(glob.glob(wap_pattern))
    main_list = []
    for member in files:
        ds = xr.open_dataset(member)
        etccdi = ds["wap"]
        
        if (etccdi["lon"].values > 180).any():
            etccdi = etccdi.assign_coords(lon=((etccdi["lon"] + 180) % 360 - 180)).sortby("lon")
            
        p2 = etccdi.sel(time=slice(str(slice_p2[0]), str(slice_p2[1]))).mean("time")
        p4 = etccdi.sel(time=slice(str(slice_p4[0]), str(slice_p4[1]))).mean("time")
        
        main_list.append((p4 - p2) * 1000.0)

    lem_main = xr.concat(main_list, dim="member").mean("member")
    
    land_area_grid = land_area_grid.interp_like(lem_main, method="nearest")
    sftlf = sftlf.interp_like(lem_main, method="nearest")

    polygon_vertices = [(-10, 65), (145, 65), (145, 20), (105, 20), (105, 50), (-10, 50)]
    lats, lons = lem_main["lat"].values, lem_main["lon"].values
    lon2d, lat2d = np.meshgrid(lons, lats)

    path = mpath.Path(polygon_vertices)
    points = np.vstack((lon2d.flatten(), lat2d.flatten())).T
    poly_mask_da = xr.DataArray(path.contains_points(points).reshape(lon2d.shape), coords={"lat": lats, "lon": lons}, dims=["lat", "lon"])

    valid_land_mask = poly_mask_da & (sftlf > 0)
    m_vals = lem_main.where(valid_land_mask, drop=True).values.flatten()
    area_vals = land_area_grid.where(valid_land_mask, drop=True).values.flatten()

    valid_idx = ~np.isnan(m_vals) & ~np.isnan(area_vals)
    m_vals, area_vals = m_vals[valid_idx], area_vals[valid_idx]

    total_land_area = np.sum(area_vals)
    m_mean = np.sum(m_vals * area_vals) / total_land_area
    m_neg_fraction = (np.sum(area_vals[m_vals < 0]) / total_land_area) * 100.0
    m_pos_fraction = (np.sum(area_vals[m_vals > 0]) / total_land_area) * 100.0

    return m_vals, area_vals, total_land_area, m_mean, m_neg_fraction, m_pos_fraction

# =============================================================================
# 2. 绘图子函数
# =============================================================================
def draw_map_subplot(ax, dataarray, sig_mask, panel_label, var_title, cmap_obj, norm_obj):
    ax.set_extent([-15, 180, 0, 90], crs=ccrs.PlateCarree())
    
    im = dataarray.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap=cmap_obj, norm=norm_obj,
        add_labels=False, add_colorbar=False
    )
    
    if sig_mask is not None:
        sig_cyc, lon_cyc = add_cyclic_point(sig_mask.values, coord=sig_mask.lon.values)
        ax.contourf(
            lon_cyc, sig_mask.lat, sig_cyc,
            levels=[-0.5, 0.5], colors='none', hatches=['///'],
            transform=ccrs.PlateCarree()
        )
        
    polygon_vertices = [(-10, 65), (145, 65), (145, 20), (105, 20), (105, 50), (-10, 50)]
    poly_eurasia = mpatches.Polygon(
        polygon_vertices, closed=True, facecolor='none',
        edgecolor='red', linewidth=1.2, linestyle='--',
        transform=ccrs.PlateCarree(), zorder=6
    )
    ax.add_patch(poly_eurasia)
    ax.coastlines(linewidth=0.5)
    
    ax.set_xticks(np.arange(0, 181, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(0, 91, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    ax.tick_params(axis='both', labelsize=9, length=2, width=0.5)
    
    ax.set_title(f"{panel_label} {var_title}", fontsize=10, fontweight='bold', loc='left')
    ax.set_title("P4-P2", fontsize=10, loc='right')
    
    return im

def draw_histogram_subplot(ax, m_vals, area_vals, total_land_area, m_mean, neg_frac, pos_frac, panel_label, right_title):
    m_min, m_max = m_vals.min(), m_vals.max()
    num_bins = 80
    bins = np.linspace(m_min, m_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    bin_width = bins[1] - bins[0]

    m_hist, _ = np.histogram(m_vals, bins=bins, weights=area_vals)
    m_land_fraction = (m_hist / total_land_area) * 100.0

    ax.bar(bin_centers, m_land_fraction, width=bin_width * 0.8, color="#7570b3", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.axvline(m_mean, color="red", linestyle="--", linewidth=1.0)

    ax.text(0.20, 0.65, f"{neg_frac:.2f}%", transform=ax.transAxes, fontsize=10, ha='center')
    ax.text(0.80, 0.7, f"{pos_frac:.2f}%", transform=ax.transAxes, fontsize=10, ha='center')
    ax.text(0.40, 0.52, "Regional Mean", transform=ax.transAxes, fontsize=9, ha='center', color='red')

    ax.set_xlim(m_min, m_max)
    base_ticks = np.arange(np.floor(m_min), np.ceil(m_max) + 1, 2)
    min_gap = (m_max - m_min) * 0.04
    filtered_ticks = [t for t in base_ticks if abs(t - m_mean) > min_gap]
    final_ticks = sorted(filtered_ticks + [m_mean])
    
   
    # ==================== 🌟 修改后的刻度设置 ====================
    ax.set_xticks(final_ticks)
    labels, tick_colors = [], []
    for t in final_ticks:
        if np.isclose(t, m_mean):
            labels.append(f"{t:.2f}")
            tick_colors.append("red")
        else:
            labels.append(f"{int(t)}" if t == int(t) else f"{t:.1f}")
            tick_colors.append("black")

    ax.set_xticklabels(labels)

    # 遍历设置颜色与对齐方式
    for label, color, t in zip(ax.get_xticklabels(), tick_colors, final_ticks):
        label.set_color(color)
        if np.isclose(t, m_mean):
            label.set_ha('right')  # 🌟 将居中改为右对齐，文字会自动整体向左偏

    ax.tick_params(axis="both", labelsize=9, length=2, width=0.5)
    ax.set_title(f"{panel_label}", fontsize=10, fontweight='bold', loc="left", pad=6)
    ax.set_title(right_title, fontsize=10, loc="right", pad=6)
    ax.set_xlabel(r"Annual $\omega$500 Change from P2 to P4 ($10^{-3}\ \mathrm{Pa\ s^{-1}}$)", 
              fontsize=9, labelpad=6, linespacing=1.5)
    ax.set_ylabel("Land Area Fraction (%)", fontsize=10, labelpad=3)
    ax.grid(False)

# =============================================================================
# 3. 主程序
# =============================================================================
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # 【时间段配置】
    # -------------------------------------------------------------------------
    ssp126_p2 = ('2065', '2085')
    ssp126_p4 = ('2280', '2300')
    
    ssp534_p2 = ('2055', '2075')
    ssp534_p4 = ('2280', '2300')

    # -------------------------------------------------------------------------
    # 【文件路径配置】
    # -------------------------------------------------------------------------
    sftlf_file = r"E:\fx\sftlf\sftlf_fx_ACCESS-ESM1-5.nc"
    areacella_file = r"E:\fx\areacella\areacella_fx_ACCESS-ESM1-5.nc"
    
    tas_path_left  = r'E:\data\tas\annual\ssp126\*.nc'
    wvc_ssp_left   = r'E:\data\prw\ssp126_year\*'
    wap_path_left  = r'E:\data\wap\500hpa\ssp126\*.nc'

    tas_path_right = r'E:\data\tas\annual\ssp534\*.nc'
    wvc_ssp_right  = r'E:\data\prw\ssp534_year\*'
    wap_path_right = r'E:\data\wap\500hpa\ssp534\*.nc'

    # -------------------------------------------------------------------------
    # 【数据计算】
    # -------------------------------------------------------------------------
    print("1/3 Calculating Row 1 (Tas)...")
    tas_lem_left, tas_agree_left   = calc_tas_hyster(tas_path_left, ssp126_p2, ssp126_p4, min_agree=26)
    tas_lem_right, tas_agree_right = calc_tas_hyster(tas_path_right, ssp534_p2, ssp534_p4, min_agree=26)

    print("2/3 Calculating Row 2 (WVC)...")
    wvc_lem_left, wvc_agree_left   = calc_wvc_cooling(wvc_ssp_left, ssp126_p2, ssp126_p4, min_agree=26)
    wvc_lem_right, wvc_agree_right = calc_wvc_cooling(wvc_ssp_right, ssp534_p2, ssp534_p4, min_agree=26)

    print("3/3 Calculating Row 3 (Omega500)...")
    m_vals_l, area_l, tot_l, mean_l, neg_l, pos_l = calc_omega_histogram(wap_path_left, sftlf_file, areacella_file, ssp126_p2, ssp126_p4)
    m_vals_r, area_r, tot_r, mean_r, neg_r, pos_r = calc_omega_histogram(wap_path_right, sftlf_file, areacella_file, ssp534_p2, ssp534_p4)

    # -------------------------------------------------------------------------
    # 【色板设置】
    # -------------------------------------------------------------------------
    tas_ticks = [-2.0, -1.6, -1.5, -1.4, -1.2, -1.1, -1.0, -0.9, -0.8, -0.6, -0.55, -0.5, -0.45, -0.4, -0.3, 0.0, 0.6]
    rdbu_base = plt.get_cmap('RdBu_r')
    colors_neg = rdbu_base(np.linspace(0.0, 0.42, 15))
    colors_pos = rdbu_base(np.linspace(0.5, 0.6, 1))
    cmap_tas = ListedColormap(np.vstack((colors_neg, colors_pos)))
    cmap_tas.set_under(colors_neg[0])
    cmap_tas.set_over(colors_pos[-1])
    norm_tas = BoundaryNorm(boundaries=tas_ticks, ncolors=len(tas_ticks) - 1)

    wvc_ticks = [-8, -6, -5, -3, -2, -1, -0.5, -0.3, -0.2, -0.1, 0, 1]
    rdbu_b = plt.get_cmap('BrBG')
    colors_n = rdbu_b(np.linspace(0.0, 0.42, 10))
    colors_p = rdbu_b(np.linspace(0.5, 0.8, 1))
    cmap_wvc = ListedColormap(np.vstack((colors_n, colors_p)))
    cmap_wvc.set_under(colors_n[0])
    cmap_wvc.set_over(colors_p[-1])
    norm_wvc = BoundaryNorm(boundaries=wvc_ticks, ncolors=len(wvc_ticks) - 1)

    # -------------------------------------------------------------------------
    # 【画布与布局配置】
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(7.5, 7))
    
    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        left=0.08, right=0.93,
        bottom=0.06, top=0.96,
        wspace=0.25, hspace=0.3
    )

    # ================= Row 1: Tas =================
    ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    im_a = draw_map_subplot(
        ax_a, tas_lem_left, tas_agree_left, '(a)', 'Tas', cmap_tas, norm_tas
        )
    im_b = draw_map_subplot(
        ax_b, tas_lem_right, tas_agree_right, '(b)', 'Tas', cmap_tas, norm_tas
        )

    # 🌟 关键修改：用 ax.text 将列标题精准画在 (a) 和 P4-P2 的正上方 (y=1.12)
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
    # ================= Row 2: WVC =================
    ax_c = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax_d = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    im_c = draw_map_subplot(ax_c, wvc_lem_left, wvc_agree_left, '(c)', 'Total Water Vapor Content', cmap_wvc, norm_wvc)
    im_d = draw_map_subplot(ax_d, wvc_lem_right, wvc_agree_right, '(d)', 'Total Water Vapor Content', cmap_wvc, norm_wvc)

    # ================= Row 3: Histograms =================
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    ax_e.set_box_aspect(90 / 195)
    ax_f.set_box_aspect(90 / 195)

    draw_histogram_subplot(ax_e, m_vals_l, area_l, tot_l, mean_l, neg_l, pos_l, '(e) ω500', 'Mid-latitude Eurasia')
    draw_histogram_subplot(ax_f, m_vals_r, area_r, tot_r, mean_r, neg_r, pos_r, '(f) ω500', 'Mid-latitude Eurasia')

    # -------------------------------------------------------------------------
    # 【精确放置 Colorbar】
    # -------------------------------------------------------------------------
    fig.canvas.draw()

    # Row 1 Colorbar
    pos_a, pos_b = ax_a.get_position(), ax_b.get_position()
    row1_left, row1_right = pos_a.x0, pos_b.x1
    row1_width = row1_right - row1_left
    cb_width = row1_width * 0.8
    cb_left = row1_left + (row1_width - cb_width) / 2
    cb_bottom_1 = pos_a.y0 - 0.06
    cb_height = 0.015

    cax1 = fig.add_axes([cb_left, cb_bottom_1, cb_width, cb_height])
    cb1 = fig.colorbar(
        im_a, cax=cax1, orientation='horizontal',
        ticks=[-1.6, -1.5, -1.4, -1.2, -1.1, -1, -0.9, -0.8, -0.6, -0.55, -0.5, -0.45, -0.4, -0.3, 0], 
        spacing='uniform', format='%g'
    )
    cb1.ax.tick_params(labelsize=8, rotation=0)
    cax1.text(
        1.03, 0.5, 
        r'K', 
        transform=cax1.transAxes, 
        fontsize=10, 
        va='center',      # 垂直居中，确保处于 colorbar 正右侧
     #   ha='left'        # 文本向右延伸
    )
    # Row 2 Colorbar
    pos_c = ax_c.get_position()
    cb_bottom_2 = pos_c.y0 - 0.06

    cax2 = fig.add_axes([cb_left, cb_bottom_2, cb_width, cb_height])
    cb2 = fig.colorbar(
        im_c, cax=cax2, orientation='horizontal',
        ticks=[-6, -5, -3, -2, -1, -0.5, -0.3, -0.2, -0.1, 0], 
        spacing='uniform', format='%g'
    )
    cb2.ax.tick_params(labelsize=8, rotation=0)
    cax2.text(
        1.03, 0.5, 
        r'mm', 
        transform=cax2.transAxes, 
        fontsize=10, 
        va='center',      # 垂直居中，确保处于 colorbar 正右侧
     #   ha='left'        # 文本向右延伸
    )
    plt.savefig('Figure04.pdf', dpi=800, bbox_inches='tight')
    
    plt.savefig('Figure04.png', dpi=800, bbox_inches='tight')
    plt.show()