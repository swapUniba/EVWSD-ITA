# EVWSD-ITA

"eval_mclip.py" is the python script to evaluate the MCLIP model we used as baseline. Add the "ds\_test\_anon\_with\_labels.json" file to the working directory when running the script. You can download it from [here](https://huggingface.co/datasets/swap-uniba/EVWSD-ITA-eval/resolve/main/ds_test_anon_with_labels.json?download=true).

"eval_submissions.py" is the python script we used to evaluate submissions sent during EVALITA 2026. The script evaluates files sent as ".csv" where each line is a ranking of file names (e.g., "1133.jpg", "850.jpg", "743.jpg", ...). If you want to evaluate submissions with a different format, adapt the script to work on your format. If you follow this format, create a "submissions" directory and drop your results file there. A "submissions_out" directory will be created with the results per instance. This script also assumes that the "ds\_test\_anon\_with\_labels.json" file is in the working directory.
