import numpy as np
from utils.acc_calculator import CorrectCalculator, MedProbCorrectCalculator, MultiHopQACorrectCalculator, RoboticPlanningCorrectCalculator
from utils.power_calculator import AutoMedProbPowerCalculator, AutoMultiHopQAPowerCalculator, AutoPowerCalculator, AutoRoboticPlanningPowerCalculator, PowerCalculator
from utils.request_tool import RequestOutput
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt, transforms
from pandas import DataFrame
import scipy
from matplotlib.patches import Ellipse

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

class PlotDrawer():
    def __init__(self):
        self.SAMPLE_INTERVAL = 30
        self.MIN_SAMPLE_SIZE = 10
        self.MAX_POWER = 2000
        

    def draw_points(self,
            data_path:str,
            enable_cot:bool,
            split_name:str,
            power_calculator:PowerCalculator,
            correct_calculator:CorrectCalculator):
        
        response_list = RequestOutput(data_path, auto_index=False)
        
        res_list = []
        
        for idx in range(len(response_list)):
            res_dict = power_calculator.calculate_power(response_list, idx, enable_cot=enable_cot,return_dict=True)
            res_dict["C"] = correct_calculator.calculate_correct(response_list, idx)
            res_dict["index"] = response_list.data[idx]["origin"]["index"]
            if res_dict["P"] > self.MAX_POWER:
                continue
            res_list.append(res_dict)
        
        p_list = {res["index"]: res['P'] for res in res_list}
        R_list = {res["index"]: res['R_CoT'] for res in res_list}
        I_list = {res["index"]: res['U_ICL'] for res in res_list}
        C_list = {res["index"]: res['C'] for res in res_list}
        middle_res = {
            "P": sum([p_list[key] for key in p_list])/len(p_list),
            "R_CoT": sum([R_list[key] for key in R_list])/len(R_list),
            "U_ICL": sum([I_list[key] for key in I_list])/len(I_list),
            "C": sum([C_list[key] for key in C_list])/len(C_list),
            }
        print(f"{split_name}\tAVG P: {middle_res['P']}")
        
        print(f"{split_name}\tAVG R_CoT: {middle_res['R_CoT']}")
        print(f"{split_name}\tAVG U_ICL: {middle_res['U_ICL']}")
        print(f"{split_name}\tAVG C: {middle_res['C']}")
        correct_list = {}
        idx_list = {}
        for res in res_list:
            key = int(res["P"]/self.SAMPLE_INTERVAL)
            if key not in correct_list:
                correct_list[key] = []
                idx_list[key] = []
            correct_list[key].append(res["C"])
            idx_list[key].append(res["index"])
        key_list = list(correct_list.keys())
        draw_data = {"acc": [], "p": [], "index": []}
        total = 0
        correct = 0
        for key in sorted(key_list):
            total += len(correct_list[key])
            correct += sum(correct_list[key])
            if len(correct_list[key]) >= self.MIN_SAMPLE_SIZE:
                draw_data["p"].append(key*self.SAMPLE_INTERVAL)
                draw_data["acc"].append(sum(correct_list[key])/len(correct_list[key]))
                draw_data["index"].append(idx_list[key])
        print("Total: ", total, " Correct: ", correct, " Acc: ", round(correct/total*100, 2))
        return draw_data, p_list, middle_res
    
    def draw(self, path_dict, save_path, specific_color=True, unified_draw=False, print_split_cor=False, draw_flag=True, draw_group=False):
        sns.set_theme(style="ticks")
        res_data = {"acc": [], "p": [], "class": []}
    
        dict_p_list = {}
        middle_res = {}
        for key in path_dict:
            # Prepare correct_calculator
            if "task_name" not in path_dict[key]:
                path_dict[key]["task_name"] = "math"
            if path_dict[key]["task_name"] == "math":
                correct_calculator = CorrectCalculator()
            elif path_dict[key]["task_name"] == "multihopqa":
                correct_calculator = MultiHopQACorrectCalculator()
            elif path_dict[key]["task_name"] == "robotic-planning":
                correct_calculator = RoboticPlanningCorrectCalculator()
            elif path_dict[key]["task_name"] == "med-prob":
                correct_calculator = MedProbCorrectCalculator()
            # Prepare power_calculator
            if path_dict[key]["task_name"] == "math":
                power_calculator = AutoPowerCalculator(
                                    reason_model=path_dict[key]["reason_model"],
                                    represent_model=path_dict[key]["represent_model"],
                                    distance_model=path_dict[key].get("distance_model", "projection_length")
                                )
            elif path_dict[key]["task_name"] == "multihopqa":
                power_calculator = AutoMultiHopQAPowerCalculator(
                                    reason_model=path_dict[key]["reason_model"],
                                    represent_model=path_dict[key]["represent_model"]
                                )
            elif path_dict[key]["task_name"] == "robotic-planning":
                power_calculator = AutoRoboticPlanningPowerCalculator(
                                    reason_model=path_dict[key]["reason_model"],
                                    represent_model=path_dict[key]["represent_model"]
                                )
            elif path_dict[key]["task_name"] == "med-prob":
                power_calculator = AutoMedProbPowerCalculator(
                                    reason_model=path_dict[key]["reason_model"],
                                    represent_model=path_dict[key]["represent_model"]
                                )
            # Annotate Data & Get Point
            draw_data, p_list, temp_middle_res = self.draw_points(
                path_dict[key]["data_path"], 
                path_dict[key]["enable_cot"],
                split_name=key,
                power_calculator=power_calculator,
                correct_calculator=correct_calculator
            )
            middle_res[key] = temp_middle_res
            # Prepare analysis data format
            dict_p_list[key] = p_list
            for i, (acc, p, idx) in enumerate(zip(draw_data["acc"], draw_data["p"], draw_data["index"])):
                res_data["acc"].append(acc)
                res_data["p"].append(p)
                if unified_draw:
                    res_data["class"].append("unified")
                else:
                    res_data["class"].append(key)
        if print_split_cor and not unified_draw:
            for key in path_dict:
                temp_acc = [x for x, y in  zip(res_data["acc"], res_data["class"]) if y == key]
                temp_p = [x for x, y in  zip(res_data["p"], res_data["class"]) if y == key]
                print(f"{key} Corr", stats.spearmanr(temp_acc, temp_p))
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(temp_acc, temp_p)
                print(f"R-squared: {r_value**2:.6f}")
        print(stats.spearmanr(res_data["acc"], res_data["p"]))
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(res_data["acc"], res_data["p"])
        print(f"R-squared: {r_value**2:.6f}")
        if draw_flag:
            # print(slope, intercept)
            if draw_group:
                fig, ax_kwargs = plt.subplots(figsize=(6, 6))
                if len(path_dict) == 4:
                    color_list = [(98,40,158), (150,73,136), (200,106,114), (239,175,83)]
                else:
                    color_list = [(98,40,158), (200,106,114), (239,175,83)]
                color_list.reverse()
                color_list = [(x/256.0, y/256.0, z/256.0) for x, y, z in color_list]
                for i, key in enumerate(path_dict):
                    confidence_ellipse(np.array([x for _i, x in enumerate(res_data["acc"]) if res_data["class"][_i] == key]), np.array([x for _i, x in enumerate(res_data["p"]) if res_data["class"][_i] == key]), ax_kwargs, n_std= 1.5,
                                alpha=0.5, facecolor=color_list[i], edgecolor=color_list[i])
                    ax_kwargs.scatter(np.array([x for _i, x in enumerate(res_data["acc"]) if res_data["class"][_i] == key]), np.array([x for _i, x in enumerate(res_data["p"]) if res_data["class"][_i] == key]), c=[color_list[i] for _i, x in enumerate(res_data["p"]) if res_data["class"][_i] == key], s=10)
            else:
                if specific_color:
                    color_list = [(98,40,158), (200,106,114), (239,175,83)]
                    color_list.reverse()
                    color_list = [(x/256.0, y/256.0, z/256.0) for x, y, z in color_list]
                    g = sns.lmplot(
                        data=DataFrame(res_data),
                        x="acc", y="p", hue="class",
                        height=5, robust=True, palette=color_list, 
                    )
                else:
                    g = sns.lmplot(
                        data=DataFrame(res_data),
                        x="acc", y="p", hue="class",
                        height=5, robust=True,
                    )
            plt.savefig(save_path, format='svg')
            plt.show()
        return middle_res