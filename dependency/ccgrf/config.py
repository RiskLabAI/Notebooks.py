# config.py

class Config:
    # General settings
    RANDOM_SEED = 42
    N_ESTIMATORS = 1000
    MIN_SAMPLES_LEAF_SIMULATION = 200
    MIN_SAMPLES_LEAF_REAL_DATA = 30
    N_BOOTSTRAP_SIMULATION = 100 # For the simulation in section 3.2, 3rd example (non-unit variance)
    N_BOOTSTRAP_REAL_DATA = 100 # Number of bootstrap samples for real data analysis

    # Simulation settings
    GAUSSIAN_N_SAMPLES = 100000
    NON_LINEAR_N_SAMPLES = 100000
    NON_LINEAR_NON_UNIT_VARIANCE_N_SAMPLES = 10000 # Adjusted for the 3rd example simulation (smaller N but more bootstrap)
    NON_LINEAR_NON_UNIT_VARIANCE_BOOTSTRAP_SAMPLES = 200 # For CI of 3rd simulation
    TEST_RANGE_GAUSSIAN = (-2, 2, 0.01) # start, end, step
    TEST_RANGE_NON_LINEAR = (0.15, 10, 0.01) # start, end, step
    TEST_RANGE_REAL_DATA = (8.7, 58.7, 0.1) # VIX range based on your figures (start, end, step)

    # Real data settings
    DATA_START_DATE = '2003-01-01'
    DATA_END_DATE = '2023-12-31'
    HEDGE_FUND_NAMES = [
        'HFRXM', # Macro Hedge Funds
        'HFRXSDV', # Systematic diversified macro hedge funds
        'HFRXMA', # Event-driven Mergers Arbitrage Hedge Fund
        'HFRXMD', # Market Directional hedge fund
        'HFRXEMN' # Equity Market Neutral strategy
    ]
    HFRX_DATA_PATH = 'data/' # Relative path where HFRX CSVs are stored

    # Plotting settings
    FIG_DIR = 'figs/'
    FONT_SIZE = 12
    FONT_FAMILY = 'Times New Roman'
    DPI = 300 # High quality figures
    GAUSSIAN_FIG_NAMES = {
        'case1': 'Fig1a_Gaussian_Case1.png', # Original Fig1a
        'case2': 'Fig1b_Gaussian_Case2.png'  # Original Fig1b
    }
    NON_LINEAR_FIG_NAMES = {
        'nonlinear_eq14': 'Fig_NL_Eq14_Table.png', # This is table output, no fig
        'nonlinear_eq15': 'Fig2_Nonlinear_Eq15.png', # Original Fig2
        'nonlinear_eq16_sim_ci': 'Fig3a_Nonlinear_Eq16_SimCI.png', # Original Fig3a
        'nonlinear_eq16_boot_ci': 'Fig3b_Nonlinear_Eq16_BootCI.png', # Original Fig3b
        'nonlinear_eq16_all_params': 'Fig4_Nonlinear_Eq16_AllParams.png' # Original Fig4
    }
    REAL_DATA_FIG_NAMES = {
        'HFRXM': 'Fig5a_HFRXM.png', # Original Fig5a
        'HFRXSDV': 'Fig5b_HFRXSDV.png', # Original Fig5b
        'HFRXMA': 'Fig6_HFRXMA.png', # Original Fig6
        'HFRXMD': 'Fig7a_HFRXMD.png', # Original Fig7a
        'HFRXEMN': 'Fig7b_HFRXEMN.png' # Original Fig7b
    }