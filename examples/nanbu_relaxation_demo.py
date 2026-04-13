from utilities import main_figure_case_bundles, run_relaxation_diagnostic


def _run_figure_bundle(bundle, deterministic: bool = False):
    histories = []
    for index, case in enumerate(bundle["cases"]):
        histories.append(
            run_relaxation_diagnostic(
                case["particle_1"],
                case["particle_2"],
                iterations=case["iterations"],
                dt=case["dt"],
                hold=index > 0,
                label_prefix=case["label_prefix"],
                rng=case["seed"] if deterministic else None,
            )
        )
    return histories


def fig4(deterministic: bool = False):
    return _run_figure_bundle(main_figure_case_bundles()[0], deterministic=deterministic)


def fig5(deterministic: bool = False):
    return _run_figure_bundle(main_figure_case_bundles()[1], deterministic=deterministic)


def fig6(deterministic: bool = False):
    return _run_figure_bundle(main_figure_case_bundles()[2], deterministic=deterministic)


def run_all_figures(deterministic: bool = False):
    histories = []
    for bundle in main_figure_case_bundles():
        histories.extend(_run_figure_bundle(bundle, deterministic=deterministic))
    return histories


if __name__ == "__main__":
    run_all_figures()
