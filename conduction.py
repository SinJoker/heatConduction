import numpy as np
from scipy.optimize import fsolve
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

# pdf
# 修正后的材料数据库（单位：W/m·K）
MATERIAL_DB = {
    "低水泥浇注料": {"lambda": lambda T: 1.5698, "default_factor": 1.0},
    "轻质浇注料": {"lambda": lambda T: 0.5016, "default_factor": 1.0},
    "炉顶密封料": {"lambda": lambda T: 0.1691, "default_factor": 1.0},
    "🧱 耐火粘土砖 (2070 kg/m³)": {
        "lambda": lambda T: 0.84 + 0.00058 * T,
        "T_max": 1300,
        "default_factor": 1.0,
    },
    "🧱 耐火粘土砖 (2100 kg/m³)": {
        "lambda": lambda T: 0.81 + 0.0006 * T,
        "T_max": 1300,
        "default_factor": 1.0,
    },
    "🧱 轻质耐火粘土砖 (1300 kg/m³)": {
        "lambda": lambda T: 0.407 + 0.000349 * T,
        "T_max": 1300,
        "default_factor": 1.0,
    },
    "🧱 轻质耐火粘土砖 (1000 kg/m³)": {
        "lambda": lambda T: 0.291 + 0.000256 * T,
        "T_max": 1300,
        "default_factor": 1.0,
    },
    "🧱 硅砖": {
        "lambda": lambda T: 0.93 + 0.000698 * T,
        "T_max": 1620,
        "default_factor": 1.0,
    },
    "🧱 半硅砖": {
        "lambda": lambda T: 0.87 + 0.00052 * T,
        "T_max": 1500,
        "default_factor": 1.0,
    },
    "🧱 镁砖": {
        "lambda": lambda T: 4.65 - 0.001745 * T,
        "T_max": 1520,
        "default_factor": 1.0,
    },
    "🧱 铬镁砖": {
        "lambda": lambda T: 1.28 + 0.000407 * T,
        "T_max": 1530,
        "default_factor": 1.0,
    },
    "🧱 碳化硅砖": {
        "lambda": lambda T: 20.9 - 10.467 * T,
        "T_max": 1700,
        "default_factor": 1.0,
    },
    "🧱 高铝砖(LZ)-65": {
        "lambda": lambda T: 2.09 + 0.001861 * T,
        "T_max": 1500,
        "default_factor": 1.0,
    },
    "🧱 高铝砖(LZ)-55": {
        "lambda": lambda T: 2.09 + 0.001861 * T,
        "T_max": 1470,
        "default_factor": 1.0,
    },
    "🧱 高铝砖(LZ)-48": {
        "lambda": lambda T: 2.09 + 0.001861 * T,
        "T_max": 1420,
        "default_factor": 1.0,
    },
    "🧱 抗渗碳砖(重质)": {
        "lambda": lambda T: 0.698 + 0.000639 * T,
        "T_max": 1400,
        "default_factor": 1.0,
    },
    "🧱 抗渗碳砖(轻质)": {
        "lambda": lambda T: 0.15 + 0.000128 * T,
        "T_max": 1400,
        "default_factor": 1.0,
    },
    "🧱 轻质耐火粘土砖(800 kg/m³)": {
        "lambda": lambda T: 0.21 + 0.0002 * T,
        "T_max": 1300,
        "default_factor": 1.0,
    },
    "🧱 轻质耐火粘土砖(600 kg/m³)": {
        "lambda": lambda T: 0.13 + 0.00023 * T,
        "T_max": 1300,
        "default_factor": 1.0,
    },
    "🧱红砖": {
        "lambda": lambda T: 0.814 + 0.000465 * T,
        "T_max": None,
        "default_factor": 1.0,
    },
    "🟫 轻质浇注料(1.4)": {
        "lambda": lambda T: 0.15 + 0.0004 * T,
        "T_max": 1150,
        "default_factor": 1.0,
    },
    "🟫 轻质浇注料(1.8)": {
        "lambda": lambda T: 0.1 + 0.0007 * T,
        "T_max": 1250,
        "default_factor": 1.0,
    },
    "🟫 重质浇注料(2.2)": {
        "lambda": lambda T: 0.45 + 0.0005 * T,
        "T_max": 1400,
        "default_factor": 1.0,
    },
    "🟫 钢筋混凝土": {"lambda": lambda T: 1.55, "T_max": None, "default_factor": 1.0},
    "🟫 泡沫混凝土": {"lambda": lambda T: 0.16, "T_max": None, "default_factor": 1.0},
    "🪟 玻璃绵": {"lambda": lambda T: 0.052, "T_max": None, "default_factor": 1.0},
    "⬜ 生石灰": {"lambda": lambda T: 0.12, "T_max": None, "default_factor": 1.0},
    "⬜ 石膏板": {"lambda": lambda T: 0.41, "T_max": None, "default_factor": 1.0},
    # 新增材料
    "⬜ 岩棉板(100 kg/m³)": {
        "lambda": lambda T: 3.2 - 0.00291 * T,
        "T_max": None,
        "default_factor": 1.0,
    },
    "⬜ 混合纤维板(Al₂O₃≥72%)(加热线收缩≤2%)(1400℃x24h)(ρ≤400 kg/m³)": {
        "lambda": lambda T: 3.05 - 0.00105 * T,
        "T_max": None,
        "default_factor": 1.0,
    },
    "⬜ 混合纤维板(Al₂O₃≥68%)(加热线收缩≤2%)(1400℃x24h)(ρ≤300 kg/m³)": {
        "lambda": lambda T: 3.05 - 0.00105 * T,
        "T_max": None,
        "default_factor": 1.0,
    },
    "☁️ 耐火纤维毯(毡)(96 kg/m³)": {
        "lambda": lambda T: 3.18 - 0.00194 * T,
        "T_max": None,
        "default_factor": 1.0,
    },
    "☁️ 耐火纤维毯(毡)(128 kg/m³)": {
        "lambda": lambda T: 3.18 - 0.00174 * T,
        "default_factor": 1.0,
    },
    "☁️ 耐火纤维毯(毡)(160 kg/m³)": {
        "lambda": lambda T: 3.17 - 0.00163 * T,
        "T_max": None,
        "default_factor": 1.0,
    },
    "☁️ 耐火纤维毯(毡)(192 kg/m³)": {
        "lambda": lambda T: 3.13 - 0.00149 * T,
        "T_max": None,
        "default_factor": 1.0,
    },
    "☁️ 耐火纤维毯(毡)(288 kg/m³)": {
        "lambda": lambda T: 3.05 - 0.00125 * T,
        "T_max": None,
        "default_factor": 1.0,
    },
    "☁️ 氧化铝纤维(Al₂O₃: 80～95%)": {
        "lambda": lambda T: 3.05 - 0.00135 * T,
        "T_max": None,
        "default_factor": 1.0,
    },
    "☁️ 莫来石纤维(Al₂O₃: 72%)": {
        "lambda": lambda T: 3.05 - 0.00135 * T,
        "T_max": None,
        "default_factor": 1.0,
    },
    "🌀 棉卷(加热线收缩≤2%)(1400℃x24h)(96 kg/m³)": {
        "lambda": lambda T: 3.05 - 0.00135 * T,
        "T_max": None,
        "default_factor": 1.0,
    },
}


# 获取对流换热系数
def get_convection_coefficient(T):
    # 对流换热系数
    CONVECTION_COEFF = {
        50: 9.95,
        55: 9.99,
        60: 10.33,
        65: 10.67,
        70: 11.02,
        75: 11.3,
        80: 11.64,
        85: 11.92,
        90: 12.21,
        95: 12.49,
        100: 12.83,
        105: 13.06,
        110: 13.34,
        115: 13.63,
        120: 13.91,
        125: 14.25,
        130: 14.48,
        135: 14.82,
        140: 15.1,
        145: 15.39,
        150: 15.67,
        155: 15.96,
        160: 16.24,
        165: 16.52,
        170: 16.81,
        175: 17.15,
        180: 17.43,
        185: 17.72,
        190: 18.0,
        195: 18.34,
        200: 18.62,
    }  # W/m²·K
    """根据温度插值计算对流换热系数"""
    temps = np.array(list(CONVECTION_COEFF.keys()))
    coeffs = np.array(list(CONVECTION_COEFF.values()))
    return np.interp(T, temps, coeffs)


def build_material_profile(layers, dx=0.001):
    """优化后的材料分布构建"""
    material_profile = []
    current_pos = 0.0
    layer_idx = 0

    total_thickness = sum(layer["thickness"] for layer in layers)
    n_nodes = int(round(total_thickness / dx)) + 1

    for i in range(n_nodes):
        pos = i * dx
        while (
            layer_idx < len(layers) - 1
            and pos >= current_pos + layers[layer_idx]["thickness"] - 1e-12
        ):
            current_pos += layers[layer_idx]["thickness"]
            layer_idx += 1

        material_profile.append(
            {
                "name": layers[layer_idx]["material"],
                "factor": layers[layer_idx].get(
                    "correction_factor",
                    MATERIAL_DB[layers[layer_idx]["material"]]["default_factor"],
                ),
            }
        )

    return material_profile, n_nodes


def corrected_solver(T_in, T_amb, layers, dx=0.001):
    """物理修正后的求解器"""
    material_profile, n_nodes = build_material_profile(layers, dx)
    positions = np.linspace(0, sum(l["thickness"] for l in layers), n_nodes)

    def residual_eq(T):
        residuals = [T[0] - T_in]  # 左边界条件

        # 内部节点处理
        for i in range(1, n_nodes - 1):
            mat_prev = material_profile[i - 1]
            mat_next = material_profile[i + 1]

            k_prev = (
                MATERIAL_DB[mat_prev["name"]]["lambda"]((T[i - 1] + T[i]) / 2)
                * mat_prev["factor"]
            )
            k_next = (
                MATERIAL_DB[mat_next["name"]]["lambda"]((T[i] + T[i + 1]) / 2)
                * mat_next["factor"]
            )
            k_interface = (
                2 * k_prev * k_next / (k_prev + k_next) if (k_prev + k_next) != 0 else 0
            )

            residuals.append(
                k_prev * (T[i - 1] - T[i]) / dx - k_interface * (T[i] - T[i + 1]) / dx
            )

        # 右边界物理修正
        mat_prev = material_profile[-2]
        T_wall = T[-1]
        T_prev = T[-2]

        # 计算导热系数时考虑温度合理性
        k = max(
            MATERIAL_DB[mat_prev["name"]]["lambda"]((T_prev + T_wall) / 2)
            * mat_prev["factor"],
            0.01,
        )

        # 对流换热系数物理约束
        h = 15 * (1 + 0.02 * max(T_wall - T_amb, 0))  # 温差非负
        h = max(h, 5.0)  # 最小对流系数约束

        residuals.append(k * (T_prev - T_wall) / dx - h * (T_wall - T_amb))

        return residuals

    # 优化初始温度场
    T_initial = np.linspace(T_in, max(T_amb + 50, 100), n_nodes)  # 保证初始温度合理
    T_initial = np.clip(T_initial, T_amb + 20, None)  # 最低温度高于环境20℃

    solution = fsolve(residual_eq, T_initial, xtol=1e-6, maxfev=5000)
    temps = solution

    # 物理合理性校验
    if np.any(temps < T_amb - 5):  # 允许5℃的数值误差
        raise ValueError("温度计算结果出现非物理值，请检查材料参数或边界条件")

    # 计算最终热流密度（右边界法）
    T_wall = temps[-1]
    h = max(15 * (1 + 0.02 * max(T_wall - T_amb, 0)), 5.0)
    q_final = h * (T_wall - T_amb)

    # 计算各层界面索引
    current_position = 0.0
    interface_indices = []
    for layer in layers:
        current_position += layer["thickness"]
        idx = int(round(current_position / dx))  # 计算节点索引
        interface_indices.append(min(idx, len(temps) - 1))  # 确保不越界

    # 构建结果数据结构
    result_data = {
        "层号": list(range(1, len(layers) + 1)),
        "材料": [layer["material"] for layer in layers],
        "厚度(mm)": [round(layer["thickness"] * 1000, 2) for layer in layers],
        "界面温度(℃)": [round(temps[idx], 2) for idx in interface_indices[:-1]]
        + [round(temps[-1], 2)],
        "热流密度(W/m²)": round(q_final, 1),
    }

    return solution, positions, result_data


def web_plot(x_meters, y, layers):
    """修正单位后的绘图函数（内部处理米到毫米转换）"""
    # 单位转换：米 -> 毫米
    x = x_meters * 1000

    fig = go.Figure()

    # 绘制温度曲线（毫米单位）
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color="#1f77b4", width=1),
            marker=dict(size=2, color="#d62728"),
            hovertemplate="位置: %{x:.1f}mm<br>温度: %{y:.1f}℃<extra></extra>",
        )
    )

    # 添加材料层背景（毫米单位计算）
    current_pos_m = 0.0  # 当前层起始位置（米）
    for i, layer in enumerate(layers):
        # 计算毫米单位边界
        start_mm = current_pos_m * 1000
        end_mm = (current_pos_m + layer["thickness"]) * 1000
        current_pos_m += layer["thickness"]

        # 添加色块
        fig.add_vrect(
            x0=start_mm,
            x1=end_mm,
            fillcolor=f"hsl({(i*60)%360}, 50%, 70%)",
            opacity=0.2,
            layer="below",
            line_width=0,
            annotation_text=f"{layer['material']}",  # 直接使用材料名称
            annotation_position="bottom",  # 修改为左下角
            annotation_font_size=12,
            annotation_textangle=-90,
        )

    # 优化毫米单位布局
    max_x = np.max(x)

    fig.update_layout(
        title=f"<b>{len(layers)}层平壁导热层温度分布曲线</b>",
        height=600,
        xaxis=dict(
            title="<b>位置 (mm)</b>",
            tickmode="linear",
            tick0=0,
            dtick=50,  # 主刻度间隔50mm
            gridcolor="rgba(150, 150, 150, 0.6)",  # 加深主网格线颜色
            gridwidth=1.2,  # 加粗主网格线
            minor=dict(
                tick0=25,  # 次刻度起始位置
                dtick=25,  # 次刻度间隔25mm
                showgrid=True,
                gridcolor="rgba(200, 200, 200, 0.4)",  # 次网格线颜色
                griddash="dot",  # 虚线样式
            ),
            range=[0, max_x],
            showline=True,
            linecolor="black",  # x轴线颜色
            mirror=True,
            ticks="outside",  # 刻度外显
        ),
        yaxis=dict(
            title="<b>温度 (℃)</b>",
            rangemode="tozero",
            dtick=100,
            gridcolor="rgba(150, 150, 150, 0.6)",  # 统一主网格线样式
            gridwidth=1.2,
            minor=dict(
                dtick=50,
                showgrid=True,
                gridcolor="rgba(200, 200, 200, 0.4)",
                griddash="dot",
            ),
            showline=True,
            linecolor="black",  # y轴线颜色
            mirror=True,
            ticks="outside",  # 刻度外显
        ),
    )
    return fig


# UI
st.title("多层平壁导热计算")
# 输入参数
st.header("边界条件")
col1, col2 = st.columns(2)
with col1:
    T_in = st.number_input("内壁温度 (℃)", min_value=0.0, value=1300.0, step=50.0)
with col2:
    T_air = st.number_input("空气温度 (℃)", min_value=0.0, value=20.0, step=5.0)
# 添加导热层
st.header("添加导热层")
layers = []
with st.expander("点击添加导热层"):
    num_layers = st.number_input("层数", min_value=1, value=1)
    for i in range(num_layers):
        st.subheader(f"第 {i+1} 层")
        thickness = (
            st.number_input(
                f"厚度 (mm) - 第 {i+1} 层", min_value=1.0, value=100.0, step=10.0
            )
            / 1000
        )  # 将mm转换为m
        material = st.selectbox(
            f"材料 - 第 {i+1} 层",
            options=list(MATERIAL_DB.keys()),
            index=min(i, len(MATERIAL_DB) - 1),
        )
        layers.append({"thickness": thickness, "material": material})

if st.button("计算"):
    if len(layers) == 0:
        st.error("请至少添加一层")
    else:
        # 计算温度分布
        temps, pos, result_data = corrected_solver(
            T_in=T_in, T_amb=T_air, layers=layers
        )
        # result_data = {
        #     "层号": list(range(1, len(layers) + 1)),
        #     "材料": [layer["material"] for layer in layers],
        #     "厚度(mm)": [round(layer["thickness"] * 1000, 2) for layer in layers],
        #     "界面温度(℃)": [round(temps[idx], 1) for idx in interface_indices[:-1]]
        #     + [round(temps[-1], 1)],
        #     "热流密度(W/m²)": round(q_final, 1),
        # }
        result_data_table = pd.DataFrame(result_data)
        result_data_table.set_index("层号", inplace=True)
        # 删除 热流密度(W/m²)
        result_data_table.drop("热流密度(W/m²)", axis=1, inplace=True)
        # 显示计算结果
        st.subheader("计算结果")
        # 显示表格
        st.table(result_data_table)
        # 显示图表
        st.plotly_chart(web_plot(pos, temps, layers))
        # 显示关键参数
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "总厚度",
                f"{sum(round(layer['thickness'] * 1000, 2) for layer in layers):.2f} mm",
            )
        with col2:
            st.metric("热流密度", f"{result_data['热流密度(W/m²)']:.2f} W/m²")

        temps_df = pd.DataFrame({"位置(mm)": pos * 1000, "温度(℃)": temps})
        csv_data = temps_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="data.csv",
            mime="text/csv",
            icon=":material/download:",
        )
