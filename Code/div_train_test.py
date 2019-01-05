f=open('pre_nonmed_terms','r').read();
f=f.split('\n');
f_train=open('all_train_nonmed_terms','w');
f_test=open('all_test_nonmed_terms','w');
i=0;
l=len(f);
while i<l:
    if i%5==0:
        f_test.write(f[i]+'\n')
    else:
        f_train.write(f[i]+'\n')
    i=i+1;