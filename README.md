**Installation**

1. Clone the repository:
    ```sh
    git clone <REPOSITORY_URL>
    cd <REPOSITORY_NAME>
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:
   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```

4. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

5. Place all `.csv` files to be processed in the `data/row` folder.

6. Make the magic happen by running (from the project root):
    ```sh
    python main.py
    ```

> [!NOTE]  
> **Information:** It is possible to modify the Savitzky-Golay filter parameters as well as the segmented fitting and BIC parameters in `main.py`.

> [!WARNING]  
> **Attention:** Because of the level of calculations involved, processing all the data can take a considerable amount of time. Make sure everything is ready before running this command.

---

**Metrics and Charts**

To check the evaluation metrics (\(R^2\), adjusted \(R^2\), RMS, breakpoint locations) and visualize the results, run the `notebooks/evaluation.ipynb` notebook. The only required input is the file/profile name (without extension) that should be evaluated and displayed.  

An example profile (`BW5D_YSI_20230822`) is included in the data folder for testing the notebook.

---

**Boxplots**

> [!WARNING]
**Attention:** Remember to change the profile `LRS90D_YSI_2023082` to `LRS90D_YSI_20230827` (**ADD THE '7' AT THE END**) in `fwl_2024_TW.csv`, otherwise, you will get an error.  

1. Load all `data/rawdy` (and/or `raw`) profiles and the `fwl_2024_TW.csv` file into `data/fwl_map`.  
2. Follow the instructions in `notebooks/filter_statistics.ipynb` and execute.
