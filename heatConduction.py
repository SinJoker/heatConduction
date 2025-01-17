import streamlit as st
import numpy as np
import plotly.graph_objects as go


# 材料数据库
MATERIALS = {
    "🧱 耐火粘土砖 (2070 kg/m³)": {"k": lambda T: 0.84 + 0.00058 * T, "T_max": 1300},
    "🧱 耐火粘土砖 (2100 kg/m³)": {"k": lambda T: 0.81 + 0.0006 * T, "T_max": 1300},
    "🧱 轻质耐火粘土砖 (1300 kg/m³)": {
        "k": lambda T: 0.407 + 0.000349 * T,
        "T_max": 1300,
    },
    "🧱 轻质耐火粘土砖 (1000 kg/m³)": {
        "k": lambda T: 0.291 + 0.000256 * T,
        "T_max": 1300,
    },
    "🧱 硅砖": {"k": lambda T: 0.93 + 0.000698 * T, "T_max": 1620},
    "🧱 半硅砖": {"k": lambda T: 0.87 + 0.00052 * T, "T_max": 1500},
    "🧱 镁砖": {"k": lambda T: 4.65 - 0.001745 * T, "T_max": 1520},
    "🧱 铬镁砖": {"k": lambda T: 1.28 + 0.000407 * T, "T_max": 1530},
    "🧱 碳化硅砖": {"k": lambda T: 20.9 - 10.467 * T, "T_max": 1700},
    "🧱 高铝砖(LZ)-65": {"k": lambda T: 2.09 + 0.001861 * T, "T_max": 1500},
    "🧱 高铝砖(LZ)-55": {"k": lambda T: 2.09 + 0.001861 * T, "T_max": 1470},
    "🧱 高铝砖(LZ)-48": {"k": lambda T: 2.09 + 0.001861 * T, "T_max": 1420},
    "🧱 抗渗碳砖(重质)": {"k": lambda T: 0.698 + 0.000639 * T, "T_max": 1400},
    "🧱 抗渗碳砖(轻质)": {"k": lambda T: 0.15 + 0.000128 * T, "T_max": 1400},
    "🧱 轻质耐火粘土砖(800 kg/m³)": {"k": lambda T: 0.21 + 0.0002 * T, "T_max": 1300},
    "🧱 轻质耐火粘土砖(600 kg/m³)": {
        "k": lambda T: 0.13 + 0.00023 * T,
        "T_max": 1300,
    },
    "🧱红砖": {"k": lambda T: 0.814 + 0.000465 * T, "T_max": None},
    "🟫 轻质浇注料(1.4)": {"k": lambda T: 0.15 + 0.0004 * T, "T_max": 1150},
    "🟫 轻质浇注料(1.8)": {"k": lambda T: 0.1 + 0.0007 * T, "T_max": 1250},
    "🟫 重质浇注料(2.2)": {"k": lambda T: 0.45 + 0.0005 * T, "T_max": 1400},
    "🟫 钢筋混凝土": {"k": lambda T: 1.55, "T_max": None},
    "🟫 泡沫混凝土": {"k": lambda T: 0.16, "T_max": None},
    "🪟 玻璃绵": {"k": lambda T: 0.052, "T_max": None},
    "⬜ 生石灰": {"k": lambda T: 0.12, "T_max": None},
    "⬜ 石膏板": {"k": lambda T: 0.41, "T_max": None},
    # 新增材料
    "⬜ 岩棉板(100 kg/m³)": {"k": lambda T: 3.2 - 0.00291 * T, "T_max": None},
    "⬜ 混合纤维板(Al₂O₃≥72%)(加热线收缩≤2%)(1400℃x24h)(ρ≤400 kg/m³)": {
        "k": lambda T: 3.05 - 0.00105 * T,
        "T_max": None,
    },
    "⬜ 混合纤维板(Al₂O₃≥68%)(加热线收缩≤2%)(1400℃x24h)(ρ≤300 kg/m³)": {
        "k": lambda T: 3.05 - 0.00105 * T,
        "T_max": None,
    },
    "☁️ 耐火纤维毯(毡)(96 kg/m³)": {"k": lambda T: 3.18 - 0.00194 * T, "T_max": None},
    "☁️ 耐火纤维毯(毡)(128 kg/m³)": {
        "k": lambda T: 3.18 - 0.00174 * T,
        "T_max": None,
    },
    "☁️ 耐火纤维毯(毡)(160 kg/m³)": {
        "k": lambda T: 3.17 - 0.00163 * T,
        "T_max": None,
    },
    "☁️ 耐火纤维毯(毡)(192 kg/m³)": {
        "k": lambda T: 3.13 - 0.00149 * T,
        "T_max": None,
    },
    "☁️ 耐火纤维毯(毡)(288 kg/m³)": {
        "k": lambda T: 3.05 - 0.00125 * T,
        "T_max": None,
    },
    "☁️ 氧化铝纤维(Al₂O₃: 80～95%)": {"k": lambda T: 3.05 - 0.00135 * T, "T_max": None},
    "☁️ 莫来石纤维(Al₂O₃: 72%)": {"k": lambda T: 3.05 - 0.00135 * T, "T_max": None},
    "🌀 棉卷(加热线收缩≤2%)(1400℃x24h)(96 kg/m³)": {
        "k": lambda T: 3.05 - 0.00135 * T,
        "T_max": None,
    },
}

# 对流换热系数与温度关系
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


def get_convection_coefficient(T):
    """根据温度插值计算对流换热系数"""
    import numpy as np

    temps = np.array(list(CONVECTION_COEFF.keys()))
    coeffs = np.array(list(CONVECTION_COEFF.values()))
    return np.interp(T, temps, coeffs)


def calculate_conductivity(material, T):
    """根据材料和温度计算导热系数"""
    material_data = MATERIALS[material]
    if material_data["T_max"] is not None and T > material_data["T_max"]:
        st.warning(
            f"警告：{material} 的温度 {T}℃ 超过最大使用温度 {material_data['T_max']}℃"
        )
    return material_data["k"](T)


def calculate_temperature_profile(layers, T_in, T_air, max_iter=2025, tol=1e-3):
    """
    计算多层平壁的温度分布（包含对流换热）
    :param layers: 各层材料参数列表 [(厚度, 材料名称), ...]
    :param T_in: 内壁温度
    :param T_air: 空气温度
    :param max_iter: 最大迭代次数
    :param tol: 收敛容差
    :return: (温度分布数组, 热流密度)
    """
    import time
    from plotly import graph_objects as go

    # 创建占位符用于动画
    plot_placeholder = st.empty()
    progress_bar = st.progress(0.0)

    # 初始假设温度分布
    temperatures = list(np.linspace(T_in, T_air, len(layers) + 2))

    # 计算各层位置
    positions = list(np.cumsum([0] + [thickness for thickness, _ in layers] + [0]))

    # 将温度数组转换为列表
    temperatures = list(temperatures)
    positions = list(positions)

    # 确保所有数组都转换为列表
    conductivities = []

    for iter in range(max_iter):
        # 更新进度条
        progress = (iter + 1) / max_iter
        progress_bar.progress(progress)

        # 计算各层导热系数
        conductivities = list(
            [
                calculate_conductivity(
                    material, (temperatures[i] + temperatures[i + 1]) / 2
                )
                for i, (_, material) in enumerate(layers)
            ]
        )

        # 计算导热热阻
        conduction_resistance = sum(
            thickness / conductivity
            for (thickness, _), conductivity in zip(layers, conductivities)
        )

        # 更新温度分布
        if iter % 5 == 0 or iter == max_iter - 1:  # 每5次迭代更新一次
            plot_placeholder.text(f"正在计算... {progress*100:.0f}%")

    # 根据外壁温度计算对流换热系数
    h = get_convection_coefficient(temperatures[-2])

    # 计算对流热阻
    convection_resistance = 1 / h

    # 总热阻
    total_resistance = conduction_resistance + convection_resistance

    # 计算热流密度
    q = (T_in - T_air) / total_resistance

    # 计算各界面温度
    temperatures = [T_in]
    cumulative_resistance = 0
    for (thickness, material), conductivity in zip(layers, conductivities):
        cumulative_resistance += thickness / conductivity
        T = T_in - q * cumulative_resistance
        temperatures = list(temperatures)  # 确保是列表
        temperatures.append(T)

    # 添加空气温度
    temperatures = list(temperatures)  # 确保是列表
    temperatures.append(T_air)

    # 返回列表而不是numpy数组
    return list(temperatures), q, total_resistance


# Streamlit 应用
st.title("多层平壁导热计算")

# 输入参数
st.header("输入参数")
col1, col2 = st.columns(2)
with col1:
    T_in = st.number_input("内壁温度 (℃)", min_value=0.0, value=1050.0, step=50.0)
with col2:
    T_air = st.number_input("空气温度 (℃)", min_value=0.0, value=20.0, step=5.0)

# 添加材料层
st.header("添加导热层")
layers = []
with st.expander("添加材料层"):
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
            options=list(MATERIALS.keys()),
            index=min(i, len(MATERIALS) - 1),
        )
        layers.append((thickness, material))

# 计算并显示结果
if st.button("计算"):
    if len(layers) == 0:
        st.error("请至少添加一个材料层")
    else:
        # 计算温度分布
        temperatures, q, total_resistance = calculate_temperature_profile(
            layers, T_in, T_air
        )

        # 计算各层位置
        positions = list(
            np.cumsum([0] + [thickness * 1000 for thickness, _ in layers] + [0])
        )  # 将m转换为mm显示

        # 使用 Plotly 绘制温度分布图
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=positions,
                y=temperatures,
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(size=8, color="red"),
            )
        )
        fig.update_layout(
            title="温度分布",
            xaxis_title="位置 (mm)",
            yaxis_title="温度 (℃)",
            showlegend=False,
            template="plotly_white",
            # 网格线设置
            xaxis=dict(
                showline=True,
                mirror=True,
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
                dtick=20,  # 固定x轴间隔
            ),
            yaxis=dict(
                showline=True,
                mirror=True,
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
                dtick=50,  # 固定y轴间隔
            ),
            # 图表边框设置
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="white",
            plot_bgcolor="white",
            # hover设置
            # hovermode="x unified",
            hoverlabel=dict(
                bgcolor="blue", font_color="white", font_size=12, font_family="Arial"
            ),
            width=800,
            height=500,
        )

        # 显示图表
        st.plotly_chart(fig)

        # 显示计算结果
        st.subheader("计算结果")

        # 创建结果表格
        result_data = {
            "层号": list(range(1, len(layers) + 1)),
            "材料": [material for _, material in layers],
            "厚度(mm)": [thickness * 1000 for thickness, _ in layers],
            "界面温度(℃)": temperatures[1:-1],
        }

        # 添加汇总行
        result_data["层号"] = list(result_data["层号"])
        result_data["材料"] = list(result_data["材料"])
        result_data["厚度(mm)"] = list(result_data["厚度(mm)"])
        result_data["界面温度(℃)"] = list(result_data["界面温度(℃)"])
        # 显示表格
        st.table(result_data)

        # 显示关键参数
        col1, col2 = st.columns(2)
        with col1:
            st.metric("热流密度", f"{q:.2f} W/m²")
            import pyperclip

            pyperclip.copy(f"{q:.2f}")
        with col2:
            st.metric("总热阻", f"{total_resistance:.2f} m²·K/W")
