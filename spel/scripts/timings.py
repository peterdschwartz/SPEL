#!/usr/bin/python3
import subprocess as sp
import sys
import os
import re

def collect_timing_across_clumps(start,stride,num,run_data):
    metrics_list = ['warp_execution_efficiency']
    metrics = ','.join(metrics_list)
    for nclumps in range(start,start+stride*num,stride):
        cmd = f"nvprof --aggregate-mode off --metrics {metrics} ./clmtest.exe {nclumps}"
        output = sp.getoutput(cmd)

        output = output.split('\n')
        ct = 0
        for el in output:
            metric = re.compile(f"{metrics_list[0]}")
            match = metric.search(el)
            if(match):
                #get rid of percentage
                s = el.replace("%",'')
                #match array of values
                match_arr = re.search(r'\[.+\]',s)
                sms = match_arr.group()
                sms = sms[1:-1]  #get rid of brackets
                sms = sms.split() #make list
                f_sms = [float(x) for x in sms] #convert to float
                non_zero = [(x != 0.0) for x in f_sms]
                num_sm = sum(non_zero)
                ##now run the cmd to get the timings
        print("Getting timings")
        cmd = f"nvprof --openacc-summary-mode inclusive ./clmtest.exe {nclumps}"
        output = sp.getoutput(cmd)
        output = output.split('\n')
        for el in output:
            compute_match = re.search(r'acc_compute_construct',el)
            if(compute_match): time = el.split(); compute_time = time[1];
            enter_data_match = re.search(r'acc_enter_data@main.F90:79',el)
            if(enter_data_match):
                el = el.replace(' OpenACC (incl):','')
                time = el.split(); data_time = time[1];
        #run_data.append([str(nclumps),str(num_sm),data_time,compute_time])
        return str(nclumps),str(num_sm),data_time,compute_time

def vary_gangs_workers_vectors(mode,start,stop,stride):
    """
    This function will edit the acc loop indepent gang clause for the number of
    gangs, workers, and/or vectors.
    """
    run_data = []
    run_data.append(['nclumps','Gangs','Workers','Vecs','#SMs','Data Time','Compute Time'])
    gang = '-'; worker = '-';vec = '-';
    for num in range(start,stop,stride):
        acc_loop = 'acc parallel loop independent gang'
        file = open('main.F90','r')
        lines = file.readlines()
        file.close()
        ct = 0; line_num = 0
        for line in lines:
            match = re.search(r'%s'%acc_loop, line)
            if(match):
                line_num = ct; print(line); break #locate line incase it changes
            ct +=1
        if(line_num == 0): sys.exit("ERROR Didn't find !$acc loop")
        newline = lines[line_num]
        newline = newline.split('gang')[0] #keep first part so indenetation is same

        if 'gang' in mode:
            num_gangs = f'num_gangs({num})'
            newline   = newline+f'gang {num_gangs} vector_length(32) & \n'
            gang = str(num); vec = '32'
        if 'worker' in mode:
            num_workers = f'num_workers({num})'
            newline   = newline+f'gang {num_workers} & \n'
            worker = str(num)
        if 'vector' in mode:
            vector_length = f'vector_length({num})'
            newline = newline+f"gang {vector_length} & \n"
            vec = str(num)

        lines[line_num] = newline
        ## write to file
        with open("main.F90",'w') as f:
            f.writelines(lines)
        ## Make
        os.system('make')
        ##call vary clumps
        nclumps,num_sm,data_time,compute_time=collect_timing_across_clumps(start=3200,stride=32,num=1,run_data=run_data)

        run_data.append([nclumps,gang,worker,vec,num_sm,data_time,compute_time])

    with open('test_gang.dat','w') as f:
        for row in run_data:
            line = '\t\t'.join(row)
            f.write(f'{line}\n')

    return gang, worker, vec

def noGPUtimings(start,stride,num):
    run_data = []
    run_data.append(['nclumps','Data Time','Compute Time'])
    for nclumps in range(start,start+stride*num,stride):
        os.system(f"./clmtest.exe {nclumps} | tail -1 >> omp_timings.dat")


run_data = []
run_data.append(['nclumps','Gangs','Workers','Vecs','#SMs','Data Time','Compute Time'])

start  = 32*80*2
stride = 32*4
num = 20
noGPUtimings(start=start,stride=stride,num=num)
sys.exit(123)

gang,worker,vec = vary_gangs_workers_vectors('gang',start=32,stop=64,stride=32)
