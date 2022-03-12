import argparse
import dill
import neuronperf


def main():
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="Run a serialized Benchmarker for a given `duration`. Upon "
        "success overwrite `filename` with the updated Benchmarker",
    )
    parser.add_argument("filename", type=str, help="The serialized Benchmarker")
    parser.add_argument("duration", type=float, help="The duration of each config (seconds)")
    parser.add_argument("--model_class_name", type=str, help="The name of a model class to load")
    parser.add_argument("--model_class_file", type=str, help="Path to Python module defining model_class_name")
    args = parser.parse_args()

    try:
        # If we were provided with a model class to import before deserialization, we need
        # to handle that now. The class will be manually imported.
        if args.model_class_name and args.model_class_file:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                args.model_class_name, args.model_class_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            globals()[args.model_class_name] = getattr(module, args.model_class_name)

        # Load the benchmarker object
        with open(args.filename, "rb") as f:
            benchmarker = dill.load(f)

        # Execute the benchmarker
        result = neuronperf.benchmarking.run_benchmarker(benchmarker, args.duration)

        # Write the result back to the same file
        with open(args.filename, "wb") as f:
            dill.dump(result, f)
    except:
        # Dump traceback to a file for debugging.
        import os
        import sys
        import traceback
        from pathlib import Path

        path = Path(args.filename)
        filename = os.path.join(path.parent, "neuronperf_error_{}".format(path.stem))
        trace = "".join(traceback.format_exception(*sys.exc_info()))
        with open(filename, "wt") as err_fp:
            err_fp.write(trace)


if __name__ == "__main__":
    main()
