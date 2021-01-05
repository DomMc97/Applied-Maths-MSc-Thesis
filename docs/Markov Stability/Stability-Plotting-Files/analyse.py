"""
Edited initially by M. Tarik Altuncu on 15th July 2020 for reference on GitHub.

This code has initially been produced to plot Markov Stability analysis in Fig.3
in the below article that you may cite:
@article{altuncu_free_2019,
    title = {From free text to clusters of content in health records: an unsupervised graph partitioning approach},
    volume = {4},
    copyright = {2019 The Author(s)},
    issn = {2364-8228},
    shorttitle = {From free text to clusters of content in health records},
    url = {https://appliednetsci.springeropen.com/articles/10.1007/s41109-018-0109-9},
    doi = {10.1007/s41109-018-0109-9},
    language = {En},
    number = {1},
    journal = {Applied Network Science},
    author = {Altuncu, M. Tarik and Mayer, Erik and Yaliraki, Sophia N. and Barahona, Mauricio},
    year = {2019},
    pages = {2},
}

Edited further by Dominic McEwen August 2020 for plots used in MSc thesis.

"""


import preferences as prfr
import utils
import os
import numpy as np
from scipy.io import loadmat
from scipy import sparse
from scipy.spatial.distance import squareform
import multiprocessing as mp
from bokeh.plotting import figure, gridplot
from bokeh.models import HoverTool, Range1d, FuncTickFormatter, Span, LogAxis
from string import ascii_uppercase
from bokeh.io import export_png, export_svgs, save, show
# added by DM  to allow in notebook display of Bokeh figures.
from bokeh.io import output_notebook
from bokeh.resources import INLINE


class MSexperiment:

    def __init__(self, mat_file, save_dir, **kwargs):
        """ load .mat file from MS Matlab results,
        analyse and save on save_dir
        available kwargs keys are: 'time_scale'
        """

        self.mat_file = mat_file
        self.mat = loadmat(mat_file)

        if 'time_scale' in kwargs.keys():
            time_scale = kwargs['time_scale']
            self.scale_time(time_scale)
        else:
            self.assign_variables(scale=np.arange(0, self.mat['Time'].shape[1]))

        self.save_dir = save_dir

    def find_by_time(self, time):
        if isinstance(time, list):
            return [np.argmin(np.abs(self.Time - t)) for t in time]
        else:
            return np.argmin(np.abs(self.Time - time))

    def scale_time(self, time_scale):
        time_array = self.mat['Time'][0]
        indices = np.where(np.logical_and(time_array >= time_scale[0],
                                          time_array <= time_scale[1]))[0]
        return self.assign_variables(scale=indices)

    def assign_variables(self, scale):
        """
        runs with full scale or by `time_scale' parameter at initialisation.
        reading the partition and assigning them to object attributes
        :param scale: full scale or cropped by `time_scale'
        :return: Nothing. Just assigns the class attributes at initialisation
        """
        try:
            self.C = self.mat['C_new'][:, scale]
        except:
            self.C = self.mat['C'][:, scale]

        def replace_by_latter(former, latter):
            """
            MS code assigns independent partiton labels at each resolution.
            This method scans over two consecutive Markov time partitions
            and reassigns the partition labels of the latter in a consistent manner to the former.
            :param former: an array of N labels at a Markov time
            :param latter: an array of N labels of the next Markov time
            :return:
            """
            old_new_count = list()
            l_uqs = np.unique(latter)
            for l_uq in l_uqs:
                l_ix = np.where(latter == l_uq)
                f_uqs, f_cnts = np.unique(former[l_ix], return_counts=True)
                f_max = f_uqs[np.argmax(f_cnts)]
                old_new_count.append((f_max, l_uq, np.max(f_cnts)))
            dictionary = dict()
            for old, new, count in old_new_count:
                if old not in dictionary:
                    dictionary[old] = (new, count)
                else:
                    if count > dictionary[old][1]:
                        dictionary[old] = (new, count)
            f_uqs, f_cnts = np.unique(former, return_counts=True)
            ix = [i for i, f_uq in enumerate(f_uqs) if f_uq not in list(dictionary.keys())]
            for i, f_uq in enumerate(f_uqs[ix][np.argsort(f_cnts[ix])[::-1]]):
                dictionary[f_uq] = (np.max([v[0] for v in list(dictionary.values())]) + 1, f_cnts[ix][i])
            return [dictionary[x][0] for x in former]

        for i in np.arange(-1, -self.C.shape[1], -1):
            latter = self.C[:, i]
            former = self.C[:, i - 1]
            self.C[:, i - 1] = replace_by_latter(former, latter)

        self.N = self.mat['N'][0][scale]
        self.S = self.mat['S'][0][scale]
        self.VI = self.mat['VI'][0][scale]
        self.Time = self.mat['Time'][0][scale]

    @staticmethod
    def plot_vitt(vi_mat, h):

        hm_data = (vi_mat - vi_mat.mean()) / vi_mat.std()
        hm_data = hm_data + np.abs(hm_data.min())
        hm_data = np.exp(1 - hm_data)
        hm_data = (hm_data / hm_data.max())

        # assign alpha values accordingly
        M, N = hm_data.shape
        img = np.empty((M, N), dtype=np.uint32)
        view = img.view(dtype=np.uint8).reshape((M, N, 4))
        for i in range(M):
            for j in range(N):
                view[i, j, 0] = 200
                view[i, j, 1] = 150
                view[i, j, 2] = 100
                view[M - i - 1, j, 3] = int(hm_data[i, j] * 255)

        def fake_color_bar(vi_mat, hm_data, w, h):
            """
            It looks ugly.
            :param vi_mat:
            :param hm_data:
            :param w:
            :param h:
            :param hide_axes:
            :return:
            """
            scale = np.logspace(hm_data.min(), hm_data.max(), hm_data.shape[0])
            brightness = np.exp(1 - scale)
            color_bar = np.repeat(np.matrix(brightness).T, hm_data.shape[1], axis=1)

            # assign alpha values accordingly
            M, N = hm_data.shape
            img = np.empty((M, N), dtype=np.uint32)
            view = img.view(dtype=np.uint8).reshape((M, N, 4))
            for i in range(M):
                for j in range(N):
                    view[i, j, 0] = 200
                    view[i, j, 1] = 150
                    view[i, j, 2] = 100
                    view[M - i - 1, j, 3] = int(color_bar[i, j] * 255)

            cb = figure(title=None, plot_width=w, plot_height=h,
                        x_range=(0, w), y_range=(vi_mat.min(), vi_mat.max()))
            cb.image_rgba(image=[img], x=0, y=vi_mat.min(), dw=w, dh=vi_mat.max())

            "Hiding axis labels etc. because this is not an actual plot."
            cb.toolbar.logo = None
            cb.toolbar_location = None
            cb.tools.visible = False
            cb.xaxis.visible = False
            cb.yaxis.visible = False

            return cb

        color_bar = fake_color_bar(vi_mat, hm_data, 100, h)
        return img, color_bar

    def plot_stability(self, h=400, vitt=True, color_bar=False, legend=False, mt_list=[]):
        """there is bug on both linear and log axis on the same figure, so we applied a workaround
            https://github.com/bokeh/bokeh/issues/7217"""
        w = int(20 * (h / 9))


        # DM removed old hover tool
        p1 = figure(x_axis_type="log", y_axis_type="log", title="Markov Stability Plot",
                    plot_width=w, plot_height=h,tools=['save'])

        if vitt:
            vi, vi_mat = varinfo(self.C.T, False)
            img, cb = self.plot_vitt(vi_mat, h)
            p1.image_rgba(image=[img], x=np.amin(self.Time), y=np.amin(self.N),
                          dw=np.amax(self.Time), dh=np.amax(self.N),
                          y_range_name="clusters")

        range_array = np.power(10, self.VI)
        p1.y_range = Range1d(np.amin(range_array), np.amax(range_array))
        p1.grid.grid_line_alpha = 0.3
        p1.xaxis.axis_label = 'Markov Time'
        p1.yaxis.axis_label = 'Variation of Information'
        p1.yaxis.axis_label_text_color = "blue"

 
        # DM added seperate inline hover tool for VI plot.
        plot1 = p1.line(self.Time, np.power(10, self.VI), color='blue',legend_label='Variation of Information')
        p1.add_tools(HoverTool(renderers=[plot1],tooltips=[('Markov time: ','@x'),('VI: ','@y')],mode='vline'))

        p1.yaxis.formatter = FuncTickFormatter(code="""
            return Math.log(tick).toFixed(2)
        """)

        range_array = self.N
        p1.extra_y_ranges["clusters"] = Range1d(np.amin(range_array), np.amax(range_array))
        p1.add_layout(
            LogAxis(y_range_name="clusters", axis_label="Number of Clusters", axis_label_text_color='red'), 'right')
        p1.x_range = Range1d(np.amin(self.Time), np.amax(self.Time))

        # DM added seperate inline hover tool for number of clusters plot.
        plot2 = p1.line(self.Time, self.N, color='red', legend_label='Number of Clusters', y_range_name="clusters")
        p1.add_tools(HoverTool(renderers=[plot2],tooltips=[('Markov time: ','@x'),('Number of Clusters: ','@y')],mode='vline'))

        if legend:
            p1.legend.location = "top_right"
        else:
            p1.legend.visible = False

        # DM changed the annotation code to improve visual appearance.
        for col_ix, mt in enumerate(mt_list):
            span = Span(location=mt, dimension='height', line_color='green', line_dash='dashed',
                            line_width=1.5)
            p1.add_layout(span)

            data_ix = self.find_by_time(mt)
            p1.text(x=[mt], x_offset=0.5, y=[np.power(10, np.max(self.VI))], y_offset=40,
                    text=[ascii_uppercase[col_ix]],
                    angle=0,
                    text_color='green', text_font_style='bold'
                    )

        if color_bar:
            return [[p1, cb]]
        else:
            return [[p1]]

    def plot(self, stability=True, vitt=True, legend=False, display=True,mt_list=[]):
        """sum up bokeh plots"""

        plots = list()

        if stability:
            plots.extend(self.plot_stability(vitt=vitt, legend=legend, mt_list=mt_list))

        p = gridplot(plots)

        self.plot_dir = os.path.join(self.save_dir, f"MS_plot")

      
        # DM added the option for inline display of plot in notebook.
        if display:
            output_notebook(INLINE)
            show(p)

        save(p, filename=f"{self.plot_dir}.html", title="Markov Stability Analysis Plot")
        export_png(p, filename=f"{self.plot_dir}.png")
        for row in plots:
            for column in row:
                column.output_backend = "svg"
        export_svgs(p, filename=f"{self.plot_dir}.svg")
        return plots


def varinfo(partition_vectors, ComputeParallel=False):
    """%VARINFO      Calculates the variation of information matrix and average
    %             between all pairs of a set of partitions
    %
    %   [VI,VI_MAT] = VARINFO(P) calculates the variation of information between
    %   each pair of partitions contained in P, where P is the N by M matrix
    %   of partitions where N is the number of nodes in the original graph
    %   and M is the number of partitions. The output VI is the average variation
    %   of information between all pairs of partitions, and VI_MAT is the M by M
    %   matrix where entry (i,j) is the variation of information between
    %   the partitions contained in column i and j of the matrix P.
    %
    %   [VI,VI_MAT] = VARINFO(P,F) allows the calculation of the variation of
    %   information in parallel if the boolean F is true and provided that
    %   matlab pool is running.
    %
    %   This code has been adapted from the code originally implemented for the
    %   following paper:
    %
    %       The performance of modularity maximization in practical contexts.
    %       B. H. Good, Y.-A. de Montjoye and A. Clauset.
    %       Physical Review E 81, 046106 (2010).
    %
    %   The original code can be found at:
    %   http://tuvalu.santafe.edu/~aaronc/modularity/"""

    number_of_partitions, n = partition_vectors.shape

    """% If all the partitions are identical, vi=0 and there is no need to do the
    % rest of the calculations which are computationally expensive."""
    a = partition_vectors[:, 0]
    if np.all(a == partition_vectors[0, :], axis=0):
        print('all partitions are identical. return 0')
        vi_mat = np.zeros((number_of_partitions, number_of_partitions))
        vi = 0
        return vi, vi_mat

    """% Select only the partitions which are different"""
    partition_vectors, b, c = np.unique(partition_vectors, return_index=True, return_inverse=True, axis=0)

    number_of_partitions = len(b)
    vi_mat = np.zeros((number_of_partitions, number_of_partitions))
    nodes = np.arange(n)

    def varinfo_column(i):
        nonlocal partition_vectors, nodes  # , n, vi, vi_mat#, vi_tot

        partition_1 = partition_vectors[i, :]
        A_1 = sparse.coo_matrix(([1] * len(nodes), (partition_1, nodes)))
        A_1.eliminate_zeros()
        n_1_all = A_1.sum(axis=1)
        vi_mat_row = np.append(i, np.zeros(number_of_partitions))  # vi_mat[i,:]
        for j in range(i):
            partition_2 = partition_vectors[j, :]
            A_2 = sparse.coo_matrix(([1] * len(nodes), (nodes, partition_2)))
            A_2.eliminate_zeros()
            n_2_all = A_2.sum(axis=0).T
            n_12_all = A_1 * A_2
            [rows, cols] = np.nonzero(n_12_all)
            n_12 = n_12_all[rows, cols].T

            n_1 = n_1_all[rows]
            n_2 = n_2_all[cols]
            vi = np.divide(np.power(n_12, 2), np.multiply(n_1, n_2))
            vi = np.sum(np.multiply(n_12, np.log(vi)))
            vi = -1 / (n * np.log(n)) * vi
            vi_mat_row[j + 1] = vi

        return vi_mat_row

    if ComputeParallel:
        try:
            cores = (min(8, mp.cpu_count(), number_of_partitions))
            p = mp.Pool(cores)
            vi_mat_row_list = p.map(varinfo_column, [i for i in range(number_of_partitions)])
            p.close()
        except Exception as e:
            print(e)
            p.close()
            raise

    else:
        vi_mat_row_list = list()
        for i in range(number_of_partitions):
            vi_mat_row_list.append(varinfo_column(i))

    for vi_mat_row in vi_mat_row_list:
        row = int(vi_mat_row[0])
        vi_mat[row, :] = vi_mat_row[1:]

    vi_mat_full = np.zeros((number_of_partitions, len(c)))

    for i in range(number_of_partitions):
        vi_mat_full[i, :] = vi_mat[i, c]

    vi_mat_full = vi_mat_full[c, :]

    vi_mat = vi_mat_full + vi_mat_full.T

    vi = np.mean(squareform(vi_mat))

    return vi, vi_mat

# new class added by DM
class SmoothMSexperiment:

    def __init__(self, mat_file, save_dir, **kwargs):
        """ load .mat file from MS Matlab results,
        analyse and save on save_dir
        available kwargs keys are: 'time_scale'
        """

        self.mat_file = mat_file
        self.mat = loadmat(mat_file)

        if 'time_scale' in kwargs.keys():
            time_scale = kwargs['time_scale']
            self.scale_time(time_scale)
        else:
            self.assign_variables(scale=np.arange(0, self.mat['Time'].shape[1]))

        self.save_dir = save_dir

    def find_by_time(self, time):
        if isinstance(time, list):
            return [np.argmin(np.abs(self.Time - t)) for t in time]
        else:
            return np.argmin(np.abs(self.Time - time))

    def scale_time(self, time_scale):
        time_array = self.mat['Time'][0]
        indices = np.where(np.logical_and(time_array >= time_scale[0],
                                          time_array <= time_scale[1]))[0]
        return self.assign_variables(scale=indices)

    def assign_variables(self, scale):
        """
        runs with full scale or by `time_scale' parameter at initialisation.
        reading the partition and assigning them to object attributes
        :param scale: full scale or cropped by `time_scale'
        :return: Nothing. Just assigns the class attributes at initialisation
        """
        try:
            self.C = self.mat['C_new'][:, scale]
        except:
            self.C = self.mat['C'][:, scale]

        def replace_by_latter(former, latter):
            """
            MS code assigns independent partiton labels at each resolution.
            This method scans over two consecutive Markov time partitions
            and reassigns the partition labels of the latter in a consistent manner to the former.
            :param former: an array of N labels at a Markov time
            :param latter: an array of N labels of the next Markov time
            :return:
            """
            old_new_count = list()
            l_uqs = np.unique(latter)
            for l_uq in l_uqs:
                l_ix = np.where(latter == l_uq)
                f_uqs, f_cnts = np.unique(former[l_ix], return_counts=True)
                f_max = f_uqs[np.argmax(f_cnts)]
                old_new_count.append((f_max, l_uq, np.max(f_cnts)))
            dictionary = dict()
            for old, new, count in old_new_count:
                if old not in dictionary:
                    dictionary[old] = (new, count)
                else:
                    if count > dictionary[old][1]:
                        dictionary[old] = (new, count)
            f_uqs, f_cnts = np.unique(former, return_counts=True)
            ix = [i for i, f_uq in enumerate(f_uqs) if f_uq not in list(dictionary.keys())]
            for i, f_uq in enumerate(f_uqs[ix][np.argsort(f_cnts[ix])[::-1]]):
                dictionary[f_uq] = (np.max([v[0] for v in list(dictionary.values())]) + 1, f_cnts[ix][i])
            return [dictionary[x][0] for x in former]

        for i in np.arange(-1, -self.C.shape[1], -1):
            latter = self.C[:, i]
            former = self.C[:, i - 1]
            self.C[:, i - 1] = replace_by_latter(former, latter)

        self.N = self.mat['N'][0][scale]
        self.S = self.mat['S'][0][scale]
        self.VI = self.mat['VI'][0][scale]
        self.Time = self.mat['Time'][0][scale]

    @staticmethod
    def plot_vitt(vi_mat, h):

        hm_data = (vi_mat - vi_mat.mean()) / vi_mat.std()
        hm_data = hm_data + np.abs(hm_data.min())
        hm_data = np.exp(1 - hm_data)
        hm_data = (hm_data / hm_data.max())

        # assign alpha values accordingly
        M, N = hm_data.shape
        img = np.empty((M, N), dtype=np.uint32)
        view = img.view(dtype=np.uint8).reshape((M, N, 4))
        for i in range(M):
            for j in range(N):
                view[i, j, 0] = 200
                view[i, j, 1] = 150
                view[i, j, 2] = 100
                view[M - i - 1, j, 3] = int(hm_data[i, j] * 255)

        def fake_color_bar(vi_mat, hm_data, w, h):
            """
            It looks ugly.
            :param vi_mat:
            :param hm_data:
            :param w:
            :param h:
            :param hide_axes:
            :return:
            """
            scale = np.logspace(hm_data.min(), hm_data.max(), hm_data.shape[0])
            brightness = np.exp(1 - scale)
            color_bar = np.repeat(np.matrix(brightness).T, hm_data.shape[1], axis=1)

            # assign alpha values accordingly
            M, N = hm_data.shape
            img = np.empty((M, N), dtype=np.uint32)
            view = img.view(dtype=np.uint8).reshape((M, N, 4))
            for i in range(M):
                for j in range(N):
                    view[i, j, 0] = 200
                    view[i, j, 1] = 150
                    view[i, j, 2] = 100
                    view[M - i - 1, j, 3] = int(color_bar[i, j] * 255)

            cb = figure(title=None, plot_width=w, plot_height=h,
                        x_range=(0, w), y_range=(vi_mat.min(), vi_mat.max()))
            cb.image_rgba(image=[img], x=0, y=vi_mat.min(), dw=w, dh=vi_mat.max())

            "Hiding axis labels etc. because this is not an actual plot."
            cb.toolbar.logo = None
            cb.toolbar_location = None
            cb.tools.visible = False
            cb.xaxis.visible = False
            cb.yaxis.visible = False

            return cb

        color_bar = fake_color_bar(vi_mat, hm_data, 100, h)
        return img, color_bar

    def plot_stability(self, h=400, vitt=True, color_bar=False, legend=False, mt_list=[]):
        """there is bug on both linear and log axis on the same figure, so we applied a workaround
            https://github.com/bokeh/bokeh/issues/7217"""
        w = int(20 * (h / 9))


        """
        Removed old hover tool.
        """
        p1 = figure(x_axis_type="log", y_axis_type="log", title="Markov Stability Plot",
                    plot_width=w, plot_height=h,tools=['save'])

        if vitt:
            vi, vi_mat = varinfo(self.C.T, False)
            img, cb = self.plot_vitt(vi_mat, h)
            p1.image_rgba(image=[img], x=np.amin(self.Time), y=np.amin(self.N),
                          dw=np.amax(self.Time), dh=np.amax(self.N),
                          y_range_name="clusters")

        range_array = np.power(10, self.VI)
        p1.y_range = Range1d(np.amin(range_array), np.amax(range_array))
        p1.grid.grid_line_alpha = 0.3
        p1.xaxis.axis_label = 'Markov Time'
        p1.yaxis.axis_label = 'Variation of Information'
        p1.yaxis.axis_label_text_color = "blue"

        """
        Added seperate hover tool for VI plot as follows and changed mode to inline.
        """
        """
        now plots smooth signal.
        """
        smooth_VI = smooth(self.VI)
        loc_min = np.r_[True, smooth_VI[1:] < smooth_VI[:-1]] & np.r_[smooth_VI[:-1] < smooth_VI[1:], True]


        plot1 = p1.line(self.Time, np.power(10, smooth_VI), color='blue',legend_label='Variation of Information')

        p1.yaxis.formatter = FuncTickFormatter(code="""
            return Math.log(tick).toFixed(2)
        """)

        range_array = self.N
        p1.extra_y_ranges["clusters"] = Range1d(np.amin(range_array), np.amax(range_array))
        p1.add_layout(
            LogAxis(y_range_name="clusters", axis_label="Number of Clusters", axis_label_text_color='red'), 'right')
        p1.x_range = Range1d(np.amin(self.Time), np.amax(self.Time))

        """
        Added seperate hover tool for VI plot as follows and changed mode to inline.
        """
        plot2 = p1.line(self.Time, self.N, color='red', legend_label='Number of Clusters', y_range_name="clusters")
        p1.add_tools(HoverTool(renderers=[plot2],tooltips=[('Markov time: ','@x'),('Number of Clusters: ','@y')],mode='vline'))

        plot3 = p1.circle(self.Time[loc_min],np.power(10, smooth_VI)[loc_min],color='blue')
        p1.add_tools(HoverTool(renderers=[plot3],tooltips=[('Markov time: ','@x'),('VI: ','@y')],mode='vline'))

        if legend:
            p1.legend.location = "top_right"
        else:
            p1.legend.visible = False

        """
        Edited Annotations.
        """
        for col_ix, mt in enumerate(mt_list):
            span = Span(location=mt, dimension='height', line_color='green', line_dash='dashed',
                            line_width=1.5)
            p1.add_layout(span)

            data_ix = self.find_by_time(mt)
            p1.text(x=[mt], x_offset=0.5, y=[np.power(10, np.max(self.VI))], y_offset=40,
                    text=[ascii_uppercase[col_ix]],
                    angle=0,
                    text_color='green', text_font_style='bold'
                    )

        if color_bar:
            return [[p1, cb]]
        else:
            return [[p1]]

    def plot(self, stability=True, vitt=True, legend=False, display=True,mt_list=[]):
        """sum up bokeh plots"""

        plots = list()

        if stability:
            plots.extend(self.plot_stability(vitt=vitt, legend=legend, mt_list=mt_list))

        p = gridplot(plots)

        self.plot_dir = os.path.join(self.save_dir, f"MS_plot_smooth")

        """
        Added option for inline display of plot in notebook.
        """
        if display:
            output_notebook(INLINE)
            show(p)

        save(p, filename=f"{self.plot_dir}.html", title="Markov Stability Analysis Plot")
        export_png(p, filename=f"{self.plot_dir}.png")
        for row in plots:
            for column in row:
                column.output_backend = "svg"
        export_svgs(p, filename=f"{self.plot_dir}.svg")
        return plots


def varinfo(partition_vectors, ComputeParallel=False):
    """%VARINFO      Calculates the variation of information matrix and average
    %             between all pairs of a set of partitions
    %
    %   [VI,VI_MAT] = VARINFO(P) calculates the variation of information between
    %   each pair of partitions contained in P, where P is the N by M matrix
    %   of partitions where N is the number of nodes in the original graph
    %   and M is the number of partitions. The output VI is the average variation
    %   of information between all pairs of partitions, and VI_MAT is the M by M
    %   matrix where entry (i,j) is the variation of information between
    %   the partitions contained in column i and j of the matrix P.
    %
    %   [VI,VI_MAT] = VARINFO(P,F) allows the calculation of the variation of
    %   information in parallel if the boolean F is true and provided that
    %   matlab pool is running.
    %
    %   This code has been adapted from the code originally implemented for the
    %   following paper:
    %
    %       The performance of modularity maximization in practical contexts.
    %       B. H. Good, Y.-A. de Montjoye and A. Clauset.
    %       Physical Review E 81, 046106 (2010).
    %
    %   The original code can be found at:
    %   http://tuvalu.santafe.edu/~aaronc/modularity/"""

    number_of_partitions, n = partition_vectors.shape

    """% If all the partitions are identical, vi=0 and there is no need to do the
    % rest of the calculations which are computationally expensive."""
    a = partition_vectors[:, 0]
    if np.all(a == partition_vectors[0, :], axis=0):
        print('all partitions are identical. return 0')
        vi_mat = np.zeros((number_of_partitions, number_of_partitions))
        vi = 0
        return vi, vi_mat

    """% Select only the partitions which are different"""
    partition_vectors, b, c = np.unique(partition_vectors, return_index=True, return_inverse=True, axis=0)

    number_of_partitions = len(b)
    vi_mat = np.zeros((number_of_partitions, number_of_partitions))
    nodes = np.arange(n)

    def varinfo_column(i):
        nonlocal partition_vectors, nodes  # , n, vi, vi_mat#, vi_tot

        partition_1 = partition_vectors[i, :]
        A_1 = sparse.coo_matrix(([1] * len(nodes), (partition_1, nodes)))
        A_1.eliminate_zeros()
        n_1_all = A_1.sum(axis=1)
        vi_mat_row = np.append(i, np.zeros(number_of_partitions))  # vi_mat[i,:]
        for j in range(i):
            partition_2 = partition_vectors[j, :]
            A_2 = sparse.coo_matrix(([1] * len(nodes), (nodes, partition_2)))
            A_2.eliminate_zeros()
            n_2_all = A_2.sum(axis=0).T
            n_12_all = A_1 * A_2
            [rows, cols] = np.nonzero(n_12_all)
            n_12 = n_12_all[rows, cols].T

            n_1 = n_1_all[rows]
            n_2 = n_2_all[cols]
            vi = np.divide(np.power(n_12, 2), np.multiply(n_1, n_2))
            vi = np.sum(np.multiply(n_12, np.log(vi)))
            vi = -1 / (n * np.log(n)) * vi
            vi_mat_row[j + 1] = vi

        return vi_mat_row

    if ComputeParallel:
        try:
            cores = (min(8, mp.cpu_count(), number_of_partitions))
            p = mp.Pool(cores)
            vi_mat_row_list = p.map(varinfo_column, [i for i in range(number_of_partitions)])
            p.close()
        except Exception as e:
            print(e)
            p.close()
            raise

    else:
        vi_mat_row_list = list()
        for i in range(number_of_partitions):
            vi_mat_row_list.append(varinfo_column(i))

    for vi_mat_row in vi_mat_row_list:
        row = int(vi_mat_row[0])
        vi_mat[row, :] = vi_mat_row[1:]

    vi_mat_full = np.zeros((number_of_partitions, len(c)))

    for i in range(number_of_partitions):
        vi_mat_full[i, :] = vi_mat[i, c]

    vi_mat_full = vi_mat_full[c, :]

    vi_mat = vi_mat_full + vi_mat_full.T

    vi = np.mean(squareform(vi_mat))

    return vi, vi_mat
