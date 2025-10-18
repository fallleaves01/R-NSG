## Dependencies and Compilation

This project is built using [**xmake**](https://xmake.io/#/).
xmake will automatically download and manage all required dependencies, but you need to install xmake beforehand.

> **Compiler requirement:**
> The code uses **C++20** features.
> Please ensure your compiler supports C++20 (e.g., GCC ≥ 11, Clang ≥ 13, MSVC ≥ 19.30).

After installing xmake, run the following commands in the project root directory:

```bash
xmake f -m release
xmake
```

---

## Example: Building and Querying the SIFT-Small Dataset

We provide a complete example using the **SIFT-Small** dataset to demonstrate data preparation, index construction, and query execution.

Enter the `scripts/` directory:

```bash
cd scripts
```

### 1. Data Preparation

```bash
python3 download.py
python3 run_py_api.py prepare
```

This step will automatically:

* Download the SIFT-Small dataset
* Generate attributes and query intervals
* Produce multi-threaded ground truth for queries

### 2. Index Construction

```bash
python3 run_py_api.py build
```

### 3. Query Execution

```bash
python3 run_py_api.py query
```

---

## Additional Datasets and Parameter Configuration

* **Adding more datasets:**
  Append new dataset information **after line 42** in `run_py_api.py`.
* **Adjusting parameters:**
  Modify the default parameters defined at the top of `run_py_api.py`,
  or directly call the APIs inside the script to execute custom experiments.
