# -*- coding: utf-8 -*-
"""
Test functions in the optic.plot module.

The plots are generated with the non-interactive Agg backend and inspected through the
matplotlib artists (axes, lines, collections, images).

"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from optic import plot as oplot  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive"),
    # raised by mpl_scatter_density when it draws (also filtered by optic.plot)
    pytest.mark.filterwarnings("ignore:All-NaN (slice|axis) encountered"),
]


@pytest.fixture(autouse=True)
def closeFigures():
    yield
    plt.close("all")


def qamSymbols(N=400, nModes=1, seed=0):
    rng = np.random.default_rng(seed)
    levels = np.array([-3, -1, 1, 3])
    sig = np.stack(
        [
            (rng.choice(levels, N) + 1j * rng.choice(levels, N)) / np.sqrt(10)
            + 0.05 * (rng.normal(size=N) + 1j * rng.normal(size=N))
            for _ in range(nModes)
        ],
        axis=1,
    )
    return sig[:, 0] if nModes == 1 else sig


CONST16 = np.array([a + 1j * b for a in (-3, -1, 1, 3) for b in (-3, -1, 1, 3)]) / np.sqrt(10)


class TestHelpers:
    @pytest.mark.parametrize(
        "nPlots, shape",
        [(1, (1, 1)), (2, (1, 2)), (4, (1, 4)), (5, (2, 3)), (6, (2, 3)), (7, (2, 4)), (10, (2, 5)), (11, (3, 4)), (12, (3, 4))],
    )
    def test_gridShape_holds_all_the_plots(self, nPlots, shape):
        assert oplot._gridShape(nPlots) == shape
        assert shape[0] * shape[1] >= nPlots

    def test_columns_gives_one_column_per_mode_without_copying_2D_inputs(self):
        x = np.arange(6, dtype=complex)
        assert oplot._columns(x).shape == (6, 1)

        x2 = x.reshape(3, 2)
        assert oplot._columns(x2) is x2

    def test_colormap_copy_is_transparent_below_vmin_without_changing_the_registered_one(self):
        transparent = oplot._colormap("turbo", transparentUnder=True)
        opaque = oplot._colormap("turbo")

        assert transparent(-1.0)[3] == 0  # under: transparent
        assert opaque(-1.0)[3] == 1
        assert plt.get_cmap("turbo")(-1.0)[3] == 1  # the registered colormap is untouched

    def test_figureAndAxes(self):
        fig, ax = plt.subplots()
        assert oplot._figureAndAxes(fig, ax) == (fig, ax)
        assert oplot._figureAndAxes(None, ax) == (fig, ax)  # the figure of the axes
        assert oplot._figureAndAxes(fig, None) == (fig, fig.gca())

        newFig, newAx = oplot._figureAndAxes(None, None)
        assert newFig is not fig and newAx.figure is newFig


class TestPconst:
    @pytest.mark.parametrize("pType", ["fast", "fancy"])
    @pytest.mark.parametrize("nModes", [1, 2, 4, 5, 6, 7, 8, 10, 11, 12])
    def test_modes_are_laid_out_in_a_grid(self, pType, nModes):
        x = qamSymbols(200, nModes)

        fig, ax = oplot.pconst(x, pType=pType)

        assert len(fig.axes) == nModes
        nRows, nCols = oplot._gridShape(nModes)
        assert ax.get_subplotspec().get_geometry()[:2] == (nRows, nCols)
        assert ax is fig.axes[-1]  # the axes of the last mode is returned
        titles = [a.get_title() for a in fig.axes]
        assert titles == ([""] if nModes == 1 else [f"mode {k}" for k in range(nModes)])

    @pytest.mark.parametrize("pType", ["fast", "fancy"])
    def test_axes_type_labels_and_limits(self, pType):
        fig, ax = oplot.pconst(qamSymbols(), R=1.7, pType=pType)

        assert (type(ax).__name__ == "ScatterDensityAxes") == (pType == "fancy")
        assert ax.get_xlabel() == "In-Phase (I)" and ax.get_ylabel() == "Quadrature (Q)"
        assert ax.get_aspect() in ("equal", 1.0)
        assert ax.get_xlim() == pytest.approx((-1.7, 1.7)) and ax.get_ylim() == pytest.approx((-1.7, 1.7))

        fig, ax = oplot.pconst(qamSymbols(), lim=False, pType=pType)
        assert ax.get_xlim() != (-1.25, 1.25)  # limits not forced

    def test_the_power_of_the_signal_is_normalized(self):
        fig, ax = oplot.pconst(10 * qamSymbols(), pType="fast")

        line = ax.get_lines()[0]
        assert np.mean(line.get_xdata() ** 2 + line.get_ydata() ** 2) == pytest.approx(1.0)

    def test_figsize(self):
        fig, _ = oplot.pconst(qamSymbols(), pType="fast", figsize=(7, 3))

        assert tuple(fig.get_size_inches()) == (7, 3)

    @pytest.mark.parametrize("pType", ["fast", "fancy"])
    @pytest.mark.parametrize("nModes", [1, 2, 7])
    def test_list_of_signals_are_overlapped_without_modifying_the_inputs(self, pType, nModes):
        signals = [qamSymbols(200, nModes, seed=0), 3 * qamSymbols(200, nModes, seed=1)]
        copies = [s.copy() for s in signals]

        fig, _ = oplot.pconst(signals, pType=pType)

        for original, copy in zip(signals, copies):
            np.testing.assert_array_equal(original, copy)
        for ax in fig.axes:
            overlapped = len(ax.get_lines()) if pType == "fast" else len(ax.images)
            assert overlapped == 2

    def test_invalid_plot_type_is_rejected_before_creating_a_figure(self):
        with pytest.raises(ValueError):
            oplot.pconst(qamSymbols(), pType="slow")

        assert plt.get_fignums() == []

    def test_density_colormap_is_selected(self):
        fig, ax = oplot.pconst(qamSymbols(), pType="fancy", cmap="viridis", whiteb=False)

        assert ax.images[0].get_cmap().name == "viridis"
        assert ax.images[0].get_cmap()(-1.0)[3] == 1  # no transparent background

        fig, ax = oplot.pconst(qamSymbols(), pType="fancy", cmap="viridis", whiteb=True)
        assert ax.images[0].get_cmap()(-1.0)[3] == 0


class TestColoredConstellationsAndBoundaries:
    def test_plotColoredConst_colors_the_symbols_by_the_decided_symbol(self):
        fig, ax = oplot.plotColoredConst(CONST16.copy(), CONST16, SNR=30)

        colors = plt.get_cmap("turbo")(np.linspace(0, 1, 16))
        scatter = ax.collections[0]
        np.testing.assert_allclose(scatter.get_facecolors(), colors)  # noiseless: symbol k -> color k
        assert ax.get_xlabel() == "In-Phase (I)" and ax.get_aspect() in ("equal", 1.0)

    def test_plotColoredConst_uses_the_given_axes_and_returns_their_figure(self):
        fig, ax = plt.subplots()

        fig1, ax1 = oplot.plotColoredConst(qamSymbols(), CONST16, ax=ax)
        fig2, ax2 = oplot.plotColoredConst(qamSymbols(), CONST16, fig=fig, ax=ax)

        assert ax1 is ax and ax2 is ax
        assert fig1 is fig and fig2 is fig
        assert len(ax.collections) == 2 and len(plt.get_fignums()) == 1

    def test_plotColoredConst_accepts_a_colormap_object(self):
        fig, ax = oplot.plotColoredConst(qamSymbols(), CONST16, cmap=plt.cm.viridis)

        assert len(ax.collections[0].get_facecolors()) == 400

    def test_plotDecisionBoundaries_draws_one_region_per_symbol(self):
        fig, ax = oplot.plotDecisionBoundaries(CONST16, gridStep=0.05)

        assert ax.get_xlabel() == "In-Phase (I)" and ax.get_aspect() in ("equal", 1.0)
        contour = ax.collections[0]
        assert len(contour.get_cmap()(np.linspace(0, 1, 16))) == 16
        # the region of every symbol appears in the contour levels
        assert len(contour.get_array()) >= 16

    def test_plotDecisionBoundaries_priors_move_the_boundaries(self):
        px = np.linspace(1, 8, 16)
        fig1, ax1 = oplot.plotDecisionBoundaries(CONST16, px=None, SNR=8, gridStep=0.05)
        fig2, ax2 = oplot.plotDecisionBoundaries(CONST16, px=px / px.sum(), SNR=8, gridStep=0.05)

        area = lambda ax: sum(len(p.vertices) for p in ax.collections[0].get_paths())  # noqa: E731
        assert area(ax1) != area(ax2)

    def test_plotDecisionBoundaries_uses_the_given_axes(self):
        fig, ax = plt.subplots()

        fig1, ax1 = oplot.plotDecisionBoundaries(CONST16, gridStep=0.05, ax=ax)

        assert ax1 is ax and fig1 is fig and len(plt.get_fignums()) == 1


class TestEyeDiagram:
    t = np.arange(4000)
    wave = np.cos(2 * np.pi * 0.05 * t) + 0.3 * np.sin(2 * np.pi * 0.11 * t)

    def test_fast_eye_diagram_of_a_real_signal(self):
        # the interpolated waveform has Nsamples points: n * 1024 for a full eye diagram
        figs, axes = oplot.eyediagram(self.wave, 4000, 8, 3, "fast", plotlabel=" Rx ")

        assert len(figs) == len(axes) == 1
        ax = axes[0]
        line = ax.get_lines()[0]
        assert ax.get_xlim() == (0.0, pytest.approx(3.0, abs=0.01))
        assert np.isnan(line.get_ydata()).any()  # the traces are broken at the wrap-around
        assert ax.get_title() == "Rx" and [t.get_text() for t in ax.get_legend().get_texts()] == ["Rx"]
        assert ax.get_xlabel() == "Symbol period ($T_s$)" and ax.get_ylabel() == "Amplitude"

    def test_complex_signals_have_one_eye_diagram_for_each_component(self):
        sig = self.wave + 1j * np.sin(2 * np.pi * 0.13 * self.t)

        figs, axes = oplot.eyediagram(sig, 1500, 8, 2, "fast", plotlabel="Rx")
        _, axesImag = oplot.eyediagram(sig.imag, 1500, 8, 2, "fast", plotlabel="Rx [imag]")
        _, axesReal = oplot.eyediagram(sig.real, 1500, 8, 2, "fast", plotlabel="Rx [real]")

        assert len(figs) == 2
        assert [a.get_title() for a in axes] == ["Rx [real]", "Rx [imag]"]
        for ax, ref in zip(axes, (axesReal[0], axesImag[0])):
            for line, refLine in zip(ax.get_lines(), ref.get_lines()):
                np.testing.assert_allclose(line.get_ydata(), refLine.get_ydata())

        # the imaginary part is not a flat line
        yImag = axes[1].get_lines()[0].get_ydata()
        assert np.nanmax(yImag) - np.nanmin(yImag) > 1.0

    def test_fancy_eye_diagram_is_a_histogram_image(self):
        figs, axes = oplot.eyediagram(self.wave, 4000, 8, 3, "fancy", plotlabel="sig")

        image = axes[0].images[0]
        assert image.get_array().shape == (350, 350)
        assert image.get_array().max() > 0
        extent = image.get_extent()
        assert tuple(extent[:2]) == (0, 3) and extent[2] < 0 < extent[3]  # the vertical range of the plotted waveform
        assert axes[0].get_title() == "sig"

    def test_dpi(self):
        figs, _ = oplot.eyediagram(self.wave, 1000, 4, 3, "fast", dpi=60)

        assert figs[0].dpi == 60

    def test_invalid_plot_type_is_rejected_before_creating_a_figure(self):
        with pytest.raises(ValueError):
            oplot.eyediagram(self.wave, 1000, 4, 3, "slow")

        assert plt.get_fignums() == []

    def test_number_of_samples_sets_the_number_of_symbols_covered(self):
        """The eye covers about Nsamples / 1024 symbols (see the Notes of eyediagram)."""
        _, axes = oplot.eyediagram(self.wave, 1024, 8, 3, "fast")

        assert axes[0].get_xlim()[1] == pytest.approx(1.0, abs=0.01)

    def test_large_number_of_samples_per_symbol(self):
        figs, _ = oplot.eyediagram(self.wave, 1500, 2048, 2, "fast")  # 1024 // SpS = 0

        assert len(figs) == 1


class TestPSD:
    def test_labels_modes_and_limits(self):
        sig = qamSymbols(2048, 2)

        fig, ax = oplot.plotPSD(sig, Fs=64e9, Fc=193.1e12, NFFT=256, label="Rx")

        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Rx: Mode 0", "Rx: Mode 1"]
        assert ax.get_xlim() == pytest.approx((193.1e12 - 32e9, 193.1e12 + 32e9))
        assert len(ax.get_lines()) == 2

    def test_no_legend_without_label(self):
        fig, ax = oplot.plotPSD(qamSymbols(1024), NFFT=128)

        assert ax.get_legend() is None and len(ax.get_lines()) == 1

    def test_plots_over_the_given_figure_even_if_it_is_not_the_current_one(self):
        fig, ax = oplot.plotPSD(qamSymbols(1024), NFFT=128, label="A")
        other = plt.figure()  # becomes the current figure

        fig2, ax2 = oplot.plotPSD(qamSymbols(1024, seed=1), NFFT=128, fig=fig, label="B")

        assert fig2 is fig and ax2 is ax
        assert len(ax.get_lines()) == 2
        assert len(other.axes) == 0  # nothing was plotted in the current figure

    def test_empty_list_still_means_a_new_figure(self):
        fig, ax = oplot.plotPSD(qamSymbols(1024), NFFT=128, fig=[])

        assert ax.figure is fig


class TestAnimationAndColormap:
    def test_animateConstGIF_saves_a_gif(self, tmp_path):
        x = qamSymbols(1000)
        path = tmp_path / "const.gif"

        oplot.animateConstGIF(x, str(path), fram=5, inter=10, title="const", centralAxes=True)

        assert path.read_bytes()[:3] == b"GIF"
        assert plt.get_fignums() == []  # the figure is closed

    @pytest.mark.parametrize("squareAxes", [True, False])
    def test_animateConstGIF_square_axes_option(self, tmp_path, monkeypatch, squareAxes):
        aspects = []
        original = matplotlib.axes.Axes.set_aspect

        def spy(self, aspect, *args, **kwargs):
            aspects.append(aspect)
            return original(self, aspect, *args, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "set_aspect", spy)

        oplot.animateConstGIF(qamSymbols(300), str(tmp_path / "c.gif"), fram=3, squareAxes=squareAxes)

        assert ("equal" in aspects) == squareAxes

    def test_animateConstGIF_short_signals(self, tmp_path):
        # fewer symbols than frames: one frame for each symbol
        oplot.animateConstGIF(qamSymbols(4), str(tmp_path / "c.gif"), fram=10)

        assert (tmp_path / "c.gif").read_bytes()[:3] == b"GIF"

    def test_randomCmap(self):
        np.random.seed(3)
        cmap = oplot.randomCmap(nColors=20, low=0.2, high=0.8)

        assert cmap.N == 20
        assert cmap.colors.min() >= 0.2 and cmap.colors.max() <= 0.8

        np.random.seed(3)
        np.testing.assert_array_equal(oplot.randomCmap(nColors=20, low=0.2, high=0.8).colors, cmap.colors)
