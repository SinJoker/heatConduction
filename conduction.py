import numpy as np
import streamlit as st
import plotly.graph_objects as go
from fenics import IntervalMesh, FunctionSpace, DirichletBC, Constant, UserExpression, near, dx, ds, grad, dot, solve

# 导热系数
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


# 获取对流换热系数
def get_convection_coefficient(T):
    """根据温度插值计算对流换热系数"""
    temps = np.array(list(CONVECTION_COEFF.keys()))
    coeffs = np.array(list(CONVECTION_COEFF.values()))
    return np.interp(T, temps, coeffs)


# 获取导热系数
def calculate_conductivity(material, T):
    """根据材料和温度计算导热系数"""
    material_data = MATERIALS[material]
    if material_data["T_max"] is not None and T > material_data["T_max"]:
        st.warning(
            f"警告：{material} 的温度 {T}℃ 超过最大使用温度 {material_data['T_max']}℃"
        )
    return material_data["k"](T)


def calculate_temperature_profile(layers, T_in, T_air, n_points=100):
    """计算多层平壁温度分布

    参数:
        layers: 各层信息列表，每个元素为(厚度, 材料名称)
        T_in: 内壁温度 (℃)
        T_air: 空气温度 (℃)
        n_points: 离散点数

    返回:
        positions: 位置数组 (m)
        temperatures: 温度数组 (℃)
    """
    from fenics import *

    # 计算总厚度
    total_thickness = sum(l[0] for l in layers)

    # 创建有限元网格
    mesh = IntervalMesh(n_points - 1, 0, total_thickness)

    # 定义函数空间
    V = FunctionSpace(mesh, "P", 1)

    # 定义边界条件
    def boundary_left(x, on_boundary):
        return on_boundary and near(x[0], 0)

    bc_left = DirichletBC(V, Constant(T_in), boundary_left)

    # 定义变分问题
    T = Function(V)
    v = TestFunction(V)

    # 定义材料属性
    class MaterialProperties(UserExpression):
        def __init__(self, layers, **kwargs):
            super().__init__(**kwargs)
            self.layers = layers
            self.boundaries = np.cumsum([0] + [l[0] for l in layers])

        def eval(self, value, x):
            pos = x[0]
            for i, (thickness, material) in enumerate(self.layers):
                if self.boundaries[i] <= pos < self.boundaries[i + 1]:
                    k = calculate_conductivity(material, T(x))
                    value[0] = k
                    return
            value[0] = 0.0

        def value_shape(self):
            return ()

    k = MaterialProperties(layers, degree=1)

    # 定义变分形式
    h = Constant(get_convection_coefficient(T_air))
    F = k * dot(grad(T), grad(v)) * dx + h * T * v * ds(1) - h * T_air * v * ds(1)

    # 求解
    solve(F == 0, T, bc_left)

    # 获取结果
    positions = np.linspace(0, total_thickness, n_points)
    temperatures = np.array([T(x) for x in positions])

    return positions, temperatures


# UI
st.title("多层平壁导热计算")

# 计算温度分布
if layers:  # 确保有导热层
    positions, temperatures = calculate_temperature_profile(layers, T_in, T_air)

    # 绘制温度分布图
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=positions * 1000,  # 转换为mm
            y=temperatures,
            mode="lines",
            name="温度分布",
        )
    )
    fig.update_layout(
        title="温度分布曲线",
        xaxis_title="位置 (mm)",
        yaxis_title="温度 (℃)",
        showlegend=True,
    )
    st.plotly_chart(fig)

# 输入参数
st.header("输入参数")
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
            options=list(MATERIALS.keys()),
            index=min(i, len(MATERIALS) - 1),
        )
        layers.append((thickness, material))
        "T_max": None,
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


# 获取对流换热系数
def get_convection_coefficient(T):
    """根据温度插值计算对流换热系数"""
    temps = np.array(list(CONVECTION_COEFF.keys()))
    coeffs = np.array(list(CONVECTION_COEFF.values()))
    return np.interp(T, temps, coeffs)


# 获取导热系数
def calculate_conductivity(material, T):
    """根据材料和温度计算导热系数"""
    material_data = MATERIALS[material]
    if material_data["T_max"] is not None and T > material_data["T_max"]:
        st.warning(
            f"警告：{material} 的温度 {T}℃ 超过最大使用温度 {material_data['T_max']}℃"
        )
    return material_data["k"](T)


def calculate_temperature_profile(layers, T_in, T_air, n_points=100):
    """计算多层平壁温度分布

    参数:
        layers: 各层信息列表，每个元素为(厚度, 材料名称)
        T_in: 内壁温度 (℃)
        T_air: 空气温度 (℃)
        n_points: 离散点数

    返回:
        positions: 位置数组 (m)
        temperatures: 温度数组 (℃)
    """
    from fenics import *

    # 计算总厚度
    total_thickness = sum(l[0] for l in layers)

    # 创建有限元网格
    mesh = IntervalMesh(n_points - 1, 0, total_thickness)

    # 定义函数空间
    V = FunctionSpace(mesh, "P", 1)

    # 定义边界条件
    def boundary_left(x, on_boundary):
        return on_boundary and near(x[0], 0)

    bc_left = DirichletBC(V, Constant(T_in), boundary_left)

    # 定义变分问题
    T = Function(V)
    v = TestFunction(V)

    # 定义材料属性
    class MaterialProperties(UserExpression):
        def __init__(self, layers, **kwargs):
            super().__init__(**kwargs)
            self.layers = layers
            self.boundaries = np.cumsum([0] + [l[0] for l in layers])

        def eval(self, value, x):
            pos = x[0]
            for i, (thickness, material) in enumerate(self.layers):
                if self.boundaries[i] <= pos < self.boundaries[i + 1]:
                    k = calculate_conductivity(material, T(x))
                    value[0] = k
                    return
            value[0] = 0.0

        def value_shape(self):
            return ()

    k = MaterialProperties(layers, degree=1)

    # 定义变分形式
    h = Constant(get_convection_coefficient(T_air))
    F = k * dot(grad(T), grad(v)) * dx + h * T * v * ds(1) - h * T_air * v * ds(1)

    # 求解
    solve(F == 0, T, bc_left)

    # 获取结果
    positions = np.linspace(0, total_thickness, n_points)
    temperatures = np.array([T(x) for x in positions])

    return positions, temperatures


# UI
st.title("多层平壁导热计算")

# 计算温度分布
if layers:  # 确保有导热层
    positions, temperatures = calculate_temperature_profile(layers, T_in, T_air)

    # 绘制温度分布图
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=positions * 1000,  # 转换为mm
            y=temperatures,
            mode="lines",
            name="温度分布",
        )
    )
    fig.update_layout(
        title="温度分布曲线",
        xaxis_title="位置 (mm)",
        yaxis_title="温度 (℃)",
        showlegend=True,
    )
    st.plotly_chart(fig)

# 输入参数
st.header("输入参数")
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
            options=list(MATERIALS.keys()),
            index=min(i, len(MATERIALS) - 1),
        )
        layers.append((thickness, material))
