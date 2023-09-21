#!/bin/bash

cd ../..
for i in 2.8; do
 cd ${i}
  for j in 2.*; do
   cd ${j}
    cd MD
     cd pqr 
pwd
     cp ../*itp .
     cp ../box_md3.gro .
gmx_hrex trjconv -s ../box_md3.tpr -f ../box_md3.gro -n ../index.ndx -pbc whole -o md3.pdb << EOF
0
EOF
     grep -v HEX md3.pdb > pep_md3.pdb
     cp ../../../../trjs/pqr/new.top .
     gmx_hrex grompp -f ../em1.mdp -c pep_md3.pdb -p new.top -o pep.tpr -pp processed.top -maxwarn 44
     gmx_hrex editconf -f pep.tpr -o pep_connect.pdb -conect
     cp pep_connect.pdb ../../../../trjs/pdb_connect/${j}.pdb 
     rm \#*
     cd ..
    cd ..
   cd ..
  done
 cd ..
done

