# BruteForcePasswordCracker

**BruteForcePasswordCracker** is a parallel password-cracking tool that attempts to find 4-character passwords hashed with SHA-256 using brute-force search. The project compares the performance of two parallel computing paradigms: **OpenMP** for CPU-based multithreading and **CUDA** for GPU-based parallelism.

## 🔐 Project Description

The program generates all possible combinations of 4-character passwords (letters and digits), hashes each with SHA-256, and compares the result with a target hash. To speed up this process, the workload is parallelized using:

- **OpenMP**: Utilizes CPU threads to distribute password generation and checking.
- **CUDA**: Launches thousands of GPU threads to execute the same logic in parallel.

This project showcases how different parallelization approaches perform in a brute-force context.

## 💻 Technologies Used

- **C / C++**
- **OpenMP**
- **CUDA**
- **SHA-256 hashing**

## ⚙️ Concepts Demonstrated

- Brute-force search
- Parallel programming (CPU and GPU)
- Thread synchronization (critical sections, atomic operations)
- GPU memory management (host-device memory transfer)
- Performance measurement and benchmarking

## 📊 Performance Results

- **OpenMP**: Shows near-linear speedup with increasing number of CPU threads.
- **CUDA**: Achieves significant acceleration using thousands of GPU threads, although total thread time can increase due to overheads.

Detailed performance graphs are included in the project report.

## 📄 License

This project was developed for academic and educational purposes.
