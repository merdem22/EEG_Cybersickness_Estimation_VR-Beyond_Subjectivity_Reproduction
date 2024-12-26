
for patient in {0001,0002,0003,0005,0006,0007,1000,1001,1002,1003,1004,1101,1102}
do 
	for init in {BL,FN,FR,SN,SR,HH,HN}
	do
        fname="../../../datasets/juliete/.cache/$patient"_"$init.npz"
        
		if [ -f "$fname" ]
        then 
            python3 analysis.py --fname $fname
        fi
	done
done

mkdir -p all-channels
find *.npz -name '*all-channels.png' -exec cp {} all-channels/ \;