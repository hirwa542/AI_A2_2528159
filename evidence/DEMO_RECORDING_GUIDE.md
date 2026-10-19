# Required demonstration video — record yourself

The lecturer requires one continuous, unedited MP4 of no more than two minutes. This guide is not a substitute for the actual video.

1. Show a handwritten card with your full name, registration number 25/28159 and code AI-A2-8159; show the terminal clock and final commit hash.
2. Run `python run_all.py --student-id "ug2528159@ines.ac.rw" --seed 8159` and show the fingerprint/artifact output.
3. Open one metrics JSON and one plot. Explain a real result in your own words.
4. Run `python predict.py --json samples/valid.json` and then `python predict.py --json samples/invalid.json`.
5. Start the API with `uvicorn app:app --host 127.0.0.1 --port 8000`; open `/docs`, call `/health` and `/predict`, and show one invalid request returning a validation error.
6. Run `pytest -q`; finish by showing the report, personally signed declaration and terminal clock.


