# **ChromaticNumber**

## **User Guide**

### 1. **Connect to VEGA**
To connect to the VEGA cluster, use SSH with your username and OTP password:
```bash
ssh <username>@login.vega.izum.si
```

Enter your OTP password when prompted.

### 2. **Clone the Repository**

1. **Set up SSH connection to GitHub**:
   Generate an SSH key and add it to your GitHub account:
   ```bash
   ssh-keygen -t rsa -b 4096 -C "<your_email@example.com>"
   cat ~/.ssh/id_rsa.pub
   ```
   Add the public key under GitHub **Settings > SSH and GPG keys**.

2. **Clone the repository**:
   ```bash
   git clone git@github.com:gantz-thomas-gantz/ChromaticNumber.git
   cd ChromaticNumber
   ```

### 3. **Compile on the Cluster with MPI**  

```bash
module load OpenMPI/4.1.5-GCC-12.3.0
mpic++ -O3 src/parallel.cc -o my_program
```  

### 4. **Submit the Job**  

Submit the job using SLURM with the following command:  
```bash
chmod +x job.sh
sbatch ./job.sh
```  

### 5. **Monitor Job Status**  

Monitor the status of your job with the following commands:  

- Check job queue:  
  ```bash
  squeue -u <username>
  ```  
- For detailed job information:  
  ```bash
  scontrol show job <job_id>
  ```  
- To cancel a job:  
  ```bash
  scancel <job_id>
  ```  

### 6. **Check Job Output**  

Check the output log after execution:  
```bash
cat <job_id>.output
```  








