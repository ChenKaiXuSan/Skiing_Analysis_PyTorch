from .visualize_3d_results import run_visualization, visualize_npz


def batch_main() -> None:
	from .main import main

	main()

__all__ = [
	"batch_main",
	"run_visualization",
	"visualize_npz",
]
