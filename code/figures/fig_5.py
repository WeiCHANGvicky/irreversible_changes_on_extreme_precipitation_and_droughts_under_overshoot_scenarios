# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:29:24 2026

@author: WeiCH
"""

# -*- coding: utf-8 -*-
"""
Atlantic ITCZ Southward Shift Mechanism: SSP1-2.6 vs SSP5-3.4 (2x2 Layout)
Journal: Geophysical Research Letters (GRL)
Author: WeiCH
"""

import glob
import os
import warnings
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
import numpy as np
import xarray as xr

warnings.filterwarnings('ignore')

# =============================================================================
# 0. 全局出版级样式设置 (GRL 标准)
# =============================================================================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 0.8


# =============================================================================
# 1. 基础预处理与计算函数
# =============================================================================
def preprocess(ds):
    """统一经纬度名称并按气压层升序排列"""
    rename_dict = {}
    if 'longitude' in ds.coords or 'longitude' in ds.dims:
        rename_dict['longitude'] = 'lon'
    if 'latitude' in ds.coords or 'latitude' in ds.dims:
        rename_dict['latitude'] = 'lat'

    if rename_dict:
        ds = ds.rename(rename_dict)

    if 'plev' in ds.coords:
        ds = ds.sortby('plev', ascending=True)
        ds['plev'] = np.round(ds['plev'])

    return ds


def agreement_mask(da, dim='member', min_agree=26):
    """计算跨模式符号一致性 Mask"""
    pos = (da > 0).sum(dim)
    neg = (da < 0).sum(dim)
    agree = np.maximum(pos, neg)
    return agree >= min_agree


def get_atlantic_section(da, lat_range=slice(-30, 30)):
    """大西洋扇区 (60°W - 60°E) 跨 0° 经线拼接并做经向平均"""
    d1 = da.sel(lon=slice(300, 360), lat=lat_range)
    d2 = da.sel(lon=slice(0, 60), lat=lat_range)
    atl = xr.concat([d1, d2], dim='lon')
    return atl.mean('lon')


# =============================================================================
# 2. 数据获取与缓存保存（彻底解耦，保留各自原始网格）
# =============================================================================
def get_row1_data(tos_pattern, va_pattern, ua_pattern, slice_p3, slice_p4, prefix_name, min_agree=26):
    """获取第一行 SST 异常、风场异常及一致性 Mask（分网格存储缓存）"""
    sst_cache = f"{prefix_name}_sst.nc"
    wind_cache = f"{prefix_name}_wind.nc"

    if os.path.exists(sst_cache) and os.path.exists(wind_cache):
        print(f"--> Found cache! Directly loading Row 1 from: {sst_cache} & {wind_cache}")
        ds_sst = xr.open_dataset(sst_cache)
        ds_wind = xr.open_dataset(wind_cache)
        return ds_sst['sst_anomaly'],ds_wind['u_anomaly'], ds_wind['v_anomaly'], ds_wind['agree_mask']

    print(f"--> Cache not found. Calculating Row 1 data for {prefix_name} ...")

    folders_tos = sorted(glob.glob(tos_pattern))
    folders_v = sorted(glob.glob(va_pattern))
    folders_u = sorted(glob.glob(ua_pattern))

    # 1. 计算 SST 异常 (海洋网格)
    sst_list = []
    for folder in folders_tos:
        folder_name = os.path.basename(folder)
        files = sorted(glob.glob(os.path.join(folder, '*.nc')))
        ds = xr.open_mfdataset(files, combine='by_coords', use_cftime=True, preprocess=preprocess)
        tos = ds['tos'].expand_dims(member=[folder_name])
        
        p3 = tos.sel(year=slice(slice_p3[0], slice_p3[1])).mean('year')
        p4 = tos.sel(year=slice(slice_p4[0], slice_p4[1])).mean('year')
        sst_list.append(p4 - p3)

    lem_sst = xr.concat(sst_list, dim='member').mean('member')
    # 2. 计算 V 异常与一致性 (大气网格)
    v_s_list = []
    for folder in folders_v:
        folder_name = os.path.basename(folder)
        files = sorted(glob.glob(os.path.join(folder, '*.nc')))
        ds = xr.open_mfdataset(files, combine='by_coords', use_cftime=True, preprocess=preprocess)
        va = ds['va'].expand_dims(member=[folder_name])
        va_surface = va.sel(plev=100000, method='nearest')
        p3 = va_surface.sel(year=slice(slice_p3[0], slice_p3[1])).mean('year')
        p4 = va_surface.sel(year=slice(slice_p4[0], slice_p4[1])).mean('year')
        v_s_list.append(p4 - p3)

    le_v_s = xr.concat(v_s_list, dim='member')
    

    # 3. 计算 U 异常与一致性 (大气网格)
    u_list = []
    for folder in folders_u:
        folder_name = os.path.basename(folder)
        files = sorted(glob.glob(os.path.join(folder, '*.nc')))
        ds = xr.open_mfdataset(files, combine='by_coords', use_cftime=True, preprocess=preprocess)
        ua = ds['ua'].sel(plev=100000, method='nearest').expand_dims(member=[folder_name])
        p3 = ua.sel(year=slice(slice_p3[0], slice_p3[1])).mean('year')
        p4 = ua.sel(year=slice(slice_p4[0], slice_p4[1])).mean('year')
        u_list.append(p4 - p3)

    le_u = xr.concat(u_list, dim='member')

    # 计算未掩膜的平均风场及一致性 Mask
    lem_u = le_u.mean('member')
    lem_v_s = le_v_s.mean('member')

    agree_v_s = agreement_mask(le_v_s, min_agree=min_agree)
    agree_u = agreement_mask(le_u, min_agree=min_agree)
    agree_total = agree_u | agree_v_s

    # 导出 SST 数据集
    ds_sst = xr.Dataset({'sst_anomaly': lem_sst.drop_vars('member', errors='ignore')})
    ds_sst.to_netcdf(sst_cache)

    # 导出风场数据集
    ds_wind = xr.Dataset({
        'u_anomaly': lem_u.drop_vars('member', errors='ignore'),
        'v_anomaly': lem_v_s.drop_vars('member', errors='ignore'),
        'agree_mask': agree_total.drop_vars('member', errors='ignore').astype(int),
    })
    ds_wind.to_netcdf(wind_cache)

    print(f"    Saved SST cache to: {sst_cache}")
    print(f"    Saved Wind cache to: {wind_cache}")

    return ds_sst['sst_anomaly'], ds_wind['u_anomaly'], ds_wind['v_anomaly'], ds_wind['agree_mask']


def get_row2_data(va_pattern, wap_pattern, slice_p1, slice_p3, slice_p4, save_name, w_scale=100.0):
    """获取第二行大西洋垂直剖面数据（P1 气候态 Omega，P4-P3 V和W异常）"""
    if os.path.exists(save_name):
        print(f"--> Found cache! Directly loading Row 2 data from: {save_name}")
        ds = xr.open_dataset(save_name)
        return ds['p1_omega'], ds['v_profile_anomaly'], ds['w_profile_anomaly']

    print(f"--> Cache not found. Calculating Row 2 data for {save_name} ...")
    folders_v = sorted(glob.glob(va_pattern))
    folders_w = sorted(glob.glob(wap_pattern))

    v_list, color_list, w_list = [], [], []

    # 1. 计算 V 剖面异常
    for folder in folders_v:
        folder_name = os.path.basename(folder)
        files = sorted(glob.glob(os.path.join(folder, '*.nc')))
        ds = xr.open_mfdataset(files, combine='by_coords', use_cftime=True, preprocess=preprocess)
        va = ds['va'].expand_dims(member=[folder_name])
        va_atl = get_atlantic_section(va)
        p3 = va_atl.sel(year=slice(slice_p3[0], slice_p3[1])).mean('year')
        p4 = va_atl.sel(year=slice(slice_p4[0], slice_p4[1])).mean('year')
        v_list.append(p4 - p3)

    # 2. 计算 Omega 剖面 (P1 气候态 & P4-P3 异常)
    for folder in folders_w:
        folder_name = os.path.basename(folder)
        files = sorted(glob.glob(os.path.join(folder, '*.nc')))
        ds = xr.open_mfdataset(files, combine='by_coords', use_cftime=True, preprocess=preprocess)
        wap = ds['wap'].expand_dims(member=[folder_name])
        wind = get_atlantic_section(wap)

        p1 = wind.sel(year=slice(slice_p1[0], slice_p1[1])).mean('year')
        p3 = wind.sel(year=slice(slice_p3[0], slice_p3[1])).mean('year')
        p4 = wind.sel(year=slice(slice_p4[0], slice_p4[1])).mean('year')

        color_list.append(p1)
        w_list.append(p4 - p3)

    lem_v = xr.concat(v_list, dim='member').mean('member')
    lem_color = xr.concat(color_list, dim='member').mean('member')
    lem_w = xr.concat(w_list, dim='member').mean('member')

    # 单位换算: Pa -> hPa
    lem_color['plev'] = lem_color['plev'] / 100.0
    lem_w['plev'] = lem_w['plev'] / 100.0
    lem_v['plev'] = lem_v['plev'] / 100.0

    color_data = lem_color * 100.0
    w_data = lem_w * w_scale

    # 剖面对齐到相同网格
    lem_v = lem_v.interp_like(color_data, method='nearest')
    w_data = w_data.interp_like(color_data, method='nearest')

    ds_out = xr.Dataset({
        'p1_omega': color_data.drop_vars('member', errors='ignore'),
        'v_profile_anomaly': lem_v.drop_vars('member', errors='ignore'),
        'w_profile_anomaly': w_data.drop_vars('member', errors='ignore'),
    })

    ds_out.to_netcdf(save_name)
    print(f"    Successfully saved Row 2 cache to: {save_name}")

    return ds_out['p1_omega'], ds_out['v_profile_anomaly'], ds_out['w_profile_anomaly']


# =============================================================================
# 3. 绘图子函数（在绘图阶段进行 Mask 过滤）
# =============================================================================
def draw_row1_map(ax, lem_sst, lem_u, lem_v_s, agree_mask, orog, panel_label, cmap_sst, norm_sst, ref_val=0.2, show_quiver_key=False):
    """绘制 Row 1 地图：海温填色 + 动态掩膜后的 1000hPa 矢量风场"""
    ax.coastlines(linewidth=0.5)
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
    # 1. 绘制海温（使用海洋原网格）
    im = ax.pcolormesh(
        lem_sst.lon, lem_sst.lat, lem_sst,
        transform=ccrs.PlateCarree(),
        cmap=cmap_sst, norm=norm_sst, zorder=0
    )

    # 2. 在绘图函数内部：对风场施加 orog 地形掩膜与 agree 一致性掩膜
    orog_aligned = orog.interp_like(lem_u, method='nearest')
    
    # 过滤条件：模式符号一致 + 非高海拔陆地 (<= 0.1)
    u_filtered = lem_u.where(agree_mask == 1).where(orog_aligned <= 0.1).where(lem_u['lat']>-50)
    
    v_filtered = lem_v_s.where(agree_mask == 1).where(orog_aligned <= 0.1).where(lem_u['lat']>-50)
    
    
    # 3. 稀疏化风场采样绘制 Quiver
    skip = 5
    u_sampled = u_filtered.isel(lat=slice(None, None, skip), lon=slice(None, None, skip))
    v_sampled = v_filtered.isel(lat=slice(None, None, skip), lon=slice(None, None, skip))
    lon_grid, lat_grid = np.meshgrid(u_sampled.lon.values, u_sampled.lat.values)

    q = ax.quiver(
        lon_grid, lat_grid,
        u_sampled.values, v_sampled.values,
        transform=ccrs.PlateCarree(),
        color='black', scale=3.0, width=0.003,
        headwidth=3.5, headlength=4.5
    )

    if show_quiver_key:
        ax.quiverkey(
            q, X=1.12, Y=0.1, U=ref_val,
            label=f'{ref_val} m/s', labelpos='N',
            coordinates='axes', fontproperties={'size': 8}
        )

    ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-60, 91, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis='both', labelsize=8)

    ax.set_title(f"{panel_label} SST", fontsize=10, fontweight='bold', loc='left')
    ax.set_title("P4-P3", fontsize=10, loc='right')

    return im


def draw_row2_vertical(ax, color_data, lem_v, w_data, panel_label, norm_w, w_levels, show_quiver_key=False):
    """绘制 Row 2 大西洋垂直剖面 (P1 Ω 气候态填色 + P4-P3 环流矢量)"""
    lats = color_data['lat'].values
    plevs = color_data['plev'].values

    cf = ax.contourf(
        lats, plevs, color_data.values,
        levels=w_levels, cmap='RdBu_r',
        norm=norm_w
    )

    X, Y = np.meshgrid(lats, plevs)
    V_component = lem_v.values
    W_component = -w_data.values

    skip_x, skip_y = 3, 1

    Q = ax.quiver(
        X[::skip_y, ::skip_x], Y[::skip_y, ::skip_x],
        V_component[::skip_y, ::skip_x], W_component[::skip_y, ::skip_x],
        color='black', scale=1.0, width=0.0035,
        headwidth=3.5, headlength=4.5, zorder=5
    )

    if show_quiver_key:
        ax.quiverkey(
            Q, X=1.12, Y=0.1, U=0.05,
            label=r'0.05 m/s', labelpos='N',
            coordinates='axes', fontproperties={'size': 8}
        )

    ax.set_yscale('linear')
    ax.set_ylim(1000, 100)
    ax.set_yticks([1000, 850, 700, 500, 400, 300, 200, 100])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    lat_min, lat_max = float(lats.min()), float(lats.max())
    ax.set_xlim(lat_min, lat_max)
    xticks = np.arange(-30, 31, 10)
    xticklabels = [f'{abs(int(x))}°S' if x < 0 else (f'{int(x)}°N' if x > 0 else '0') for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_ylabel('Pressure (hPa)', fontsize=10)
    ax.tick_params(axis='both', labelsize=8, length=2)

    ax.set_title(f"{panel_label} ω", fontsize=10, fontweight='bold', loc='left')
    ax.set_title("P4-P3", fontsize=10, loc='right')

    return cf


# =============================================================================
# 4. 主程序
# =============================================================================
if __name__ == "__main__":

    # 1. 时间段与地形数据
    ssp126_p1, ssp126_p3, ssp126_p4 = (2035, 2055), (2170, 2190), (2280, 2300)
    ssp534_p1, ssp534_p3, ssp534_p4 = (2025, 2045), (2165, 2185), (2280, 2300)

    orog_file = r'E:\fx\orog\orog_fx_ACCESS-ESM1-5_historical_r1i1p1f1_gn.nc'
    orog = xr.open_dataset(orog_file)['orog']

    # 2. 文件路径
    tos_left = r'E:\data\tos\ssp126\*'
    va_left  = r'E:\data\va\ssp126\*'
    ua_left  = r'E:\data\ua\ssp126\*'
    wap_left = r'F:\wap\year\ssp126\*'

    tos_right = r'E:\data\tos\ssp534\*'
    va_right  = r'E:\data\va\ssp534\*'
    ua_right  = r'E:\data\ua\ssp534\*'
    wap_right = r'F:\wap\year\ssp534\*'

    # 3. 数据读取与计算缓存
    print("=== Processing Left Column (SSP1-2.6) ===")
    sst_l, u_l, v_s_l, agree_l = get_row1_data(
        tos_left, va_left, ua_left, 
        ssp126_p3, ssp126_p4, 
        prefix_name='cache_ssp126_row1'
    )
    color_l, v_l, w_l = get_row2_data(
        va_left, wap_left, 
        ssp126_p1, ssp126_p3, ssp126_p4, 
        save_name='cache_ssp126_row2.nc', w_scale=100.0
    )

    print("\n=== Processing Right Column (SSP5-3.4) ===")
    sst_r, u_r, v_s_r, agree_r = get_row1_data(
        tos_right, va_right, ua_right, 
        ssp534_p3, ssp534_p4, 
        prefix_name='cache_ssp534_row1'
    )
    color_r, v_r, w_r = get_row2_data(
        va_right, wap_right, 
        ssp534_p1, ssp534_p3, ssp534_p4, 
        save_name='cache_ssp534_row2.nc', w_scale=100.0
    )

    # 4. 色板规范
    sst_ticks = [-0.8, -0.5, -0.3, -0.2, -0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
    cmap_sst = plt.get_cmap('RdBu_r')
    norm_sst = BoundaryNorm(boundaries=sst_ticks, ncolors=cmap_sst.N)

    norm_w = TwoSlopeNorm(vcenter=0, vmin=-3.0, vmax=3.0)
    w_levels = np.linspace(-3.5, 3.5, 15)

    # 5. 绘图
    fig = plt.figure(figsize=(6, 5))

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.08, right=0.93,
        bottom=0.08, top=0.92,
        wspace=0.3, hspace=0.1
    )

    # Row 1 地图
    ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
    ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree(central_longitude=0))
    im_a = draw_row1_map(ax_a, sst_l, u_l, v_s_l, agree_l, orog, '(a)', cmap_sst, norm_sst, show_quiver_key=False)
    im_b = draw_row1_map(ax_b, sst_r, u_r, v_s_r, agree_r, orog, '(b)', cmap_sst, norm_sst, show_quiver_key=True)

    ax_a.text(0.5, 1.2, 'SSP1-2.6', transform=ax_a.transAxes, fontsize=10, fontweight='bold', ha='center', va='bottom')
    ax_b.text(0.5, 1.2, 'SSP5-3.4', transform=ax_b.transAxes, fontsize=10, fontweight='bold', ha='center', va='bottom')

    # Row 2 垂直剖面
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_c.set_box_aspect(110 / 195)
    ax_d.set_box_aspect(110/ 195)

    cf_c = draw_row2_vertical(ax_c, color_l, v_l, w_l, '(c)', norm_w, w_levels, show_quiver_key=False)
    cf_d = draw_row2_vertical(ax_d, color_r, v_r, w_r, '(d)', norm_w, w_levels, show_quiver_key=True)

    # 色标 Colorbars
    fig.canvas.draw()

    pos_a, pos_b = ax_a.get_position(), ax_b.get_position()
    row1_left, row1_right = pos_a.x0, pos_b.x1
    row1_width = row1_right - row1_left
    cb_width = row1_width * 0.8
    cb_left = row1_left + (row1_width - cb_width) / 2
    cb_bottom_1 = pos_a.y0 - 0.08
    cb_height = 0.015

    cax1 = fig.add_axes([cb_left, cb_bottom_1, cb_width, cb_height])
    cb1 = fig.colorbar(
        im_a, cax=cax1, orientation='horizontal',
        ticks=[-0.5, -0.3, -0.2, -0.1, -0.05,-0.01, 0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
        format='%g'
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
    
    pos_c = ax_c.get_position()
    cb_bottom_2 = pos_c.y0 - 0.08

    cax2 = fig.add_axes([cb_left, cb_bottom_2, cb_width, cb_height])
    cb2 = fig.colorbar(
        cf_c, cax=cax2, orientation='horizontal',
        ticks=[-3, -2, -1, 0, 1, 2, 3],
        format='%g'
    )
    cb2.ax.tick_params(labelsize=8, rotation=0)
    cax2.text(
        1.03, 0.5, 
        r'$10^{-2}\ \mathrm{Pa\ s^{-1}}$', 
        transform=cax2.transAxes, 
        fontsize=10, 
        va='center',      # 垂直居中，确保处于 colorbar 正右侧
     #   ha='left'        # 文本向右延伸
    )
    
    # 6. 保存与显示
    pdf_path = 'Figure05.pdf'
    png_path = 'Figure05.png'

    plt.savefig(pdf_path, dpi=800, bbox_inches='tight')
    plt.savefig(png_path, dpi=800, bbox_inches='tight')
    plt.show()
    print(f"\n>>> Saved figure successfully as:\n    1. {os.path.abspath(pdf_path)}\n    2. {os.path.abspath(png_path)}")

    plt.draw()
    plt.close('all')