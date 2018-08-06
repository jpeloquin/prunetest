# Third-party packages
import numpy as np
from scipy.optimize import minimize

def label_data(protocol, data, control_var):
    """Label a table of test data based on a protocol.

    The protocol is defined as a table of test segments with at least a
    column named "Time (s)" and a column with name == `control_var`.

    Future versions of this function will support an arbitrarily large
    number of control variables.

    """
    time_points = protocol["Time (s)"].values.copy()
    def f(p):
        """Fit quality metric for fitting origin point."""
        # Time ponts for comparison are drawn from the protocol rather
        # than the data because we want the number of residuals to
        # remain constant regardless of how much of the test data ends
        # up spanned by the fitted protocol segments.
        if i == 0:
            s_d = p[1:]  # duration, fit, by segment
            i0 = 0
            i1 = 2
        else:
            i0 = i - 1
            i1 = i + 1
            s_d = p  # duration, fit, by segment
        s_tf = time_points[i0] + np.cumsum(np.hstack([[0], s_d]))
        # ^ time, fit, by segment
        s_dp = np.diff(protocol["Time (s)"].values[i0:i1+1])
        # ^ duration, protocol, by segment
        tf = np.hstack([np.linspace(s_tf[j], s_tf[j+1], 10)
                        for j in range(len(s_tf)-1)])
        # ^ time, fit, dense
        tp = np.hstack([np.linspace(protocol["Time (s)"].values[i0+j],
                                    protocol["Time (s)"].values[i0+j+1],
                                    10)
                        for j in range(len(s_tf)-1)])
        # ^ time, protocol, dense
        yp = np.interp(tp, protocol["Time (s)"],  # y, protocol, dense
                       protocol[control_var])
        yd = np.interp(tf, data["Time (s)"], data[control_var])  # y, data, dense
        yf = np.interp(tf, s_tf, protocol[control_var][i0:i1+1])  # y, fit, dense
        r = np.corrcoef(yf, yd)[0,1]
        # r = np.cov(yf, yd)[0,1] / np.cov(yp, yp)[0,0]
        if np.isnan(r):
            if np.var(yf) == 0 and np.var(yd) == 0:
                reason = "The values of the control variable `{}` have zero variance in both the *provided data* and the *current protocol fit* for {} ≤ t ≤ {}."
            elif np.var(yd) == 0:
                reason = "The values of the control variable `{}` have zero variance in the *provided data* for {} ≤ t ≤ {}."
            elif np.var(yf) == 0:
                reason = "The values of the control variable `{}` have zero variance in the *current protocol fit* for {} ≤ t ≤ {}."
            else:
                reason = "The cause is unknown."
            msg = "The Pearson correlation coefficient between the provided data and the current protocol fit is undefined. " + reason.format(control_var, tf[0], tf[-1])
            raise(RuntimeError(msg))
        # Add time dilation penalty
        penalty = np.sum((abs(s_d - s_dp) / s_dp)**3.0) / len(s_dp)
        # stdout.write("r = {:.4f}  penalty = {:.4f}  ".format(r, penalty))
        # print("p = {}".format(p))
        return -r + penalty
    for i in range(len(protocol["Time (s)"])):
        if i == 0:
            p0 = np.hstack([[0], np.diff(time_points[i:i+3])])
        else:
            p0 = np.diff(protocol["Time (s)"][i-1:i+2])
        # print("\ni = {}".format(i))
        bounds = [(0, np.inf) for x in p0]  # BFGS can't use bounds
        result = minimize(f, p0, method="BFGS")
        time_points[i] = time_points[max([0, i-1])] + result['x'][0]
    return time_points
