The programs are written in Fortran-90. The double precision is adopted.

The following six files make up the package: 

allosbrac.f90
testallosbrac.f90
allosoutput
  
osbrac.f90
testosbrac.f90
osoutput

These are two independent groups of files that correspond to the two versions of the present
program. 

In one version, the subroutine producing the brackets is named ALLOSBRAC. It is contained in 
the file allosbrac.f90 along with the routines this subroutine uses. 

In the other version, the subroutine producing the brackets is named OSBRAC. It is contained
in the file osbrac.f90 along with the routines this subroutine uses. 

To test the set of programs contained in the file allosbrac.f90, one may say, e.g.,
gfortran -O2 testallosbrac.f90 allosbrac.f90. To test the set of programs contained in the 
file osbrac.f90, one may say, e.g., gfortran -O2 testosbrac.f90 osbrac.f90. The files
allosoutput and osoutput are respective output files that contain results of the tests. 
 
The files testallosbrac.f90 and testosbrac.f90 contain, respectively, only the programs
TESTALLOSBRAC and TESTOSBRAC. The tests performed are described in comment lines 
in these programs and in the accompanying CPC paper. 

The parameters that are set at the moment in the programs TESTALLOSBRAC and TESTOSBRAC  
are just those for which the results in the files allosoutput and osoutput are listed. With these
parameters, the programs run less than a second on a notebook. 
-------------------------------------------------------------------------------------------------------------------------------------------
As said above, the brackets are produced either by the subroutine ALLOSBRAC or by the
subroutine OSBRAC. The subroutines contain generous comments.
-------------------------------------------------------------------------------------------------------------------------------------------
In the case when ALLOSBRAC is employed, all the routines used to calculate the brackets, 
which are contained in the file allosbrac.f90, are as follows: the subroutine ALLOSBRAC (with
comments), the subroutines ARR, FLPHI (with comments), COEFREL (with comments), and the
function WIGMOD. 

The subroutine ALLOSBRAC calls for the ARR, COE, FLPHI, and COEFREL subroutines. FLPHI calls 
for the function WIGMOD. The meaning of the parameters of the ALLOSBRAC subroutine is
explained in comments at its beginning.
------------------------------------------------------------------------------------------------------------------------------------------
In the case when OSBRAC is employed, all the routines used to calculate the brackets, 
which are contained in the file osbrac.f90, are as follows: the subroutine OSBRAC (with
comments), the subroutines ARR, FLPHI (with comments), COEFREL (with comments), and the
function WIGMOD. 

The subroutine OSBRAC calls for  the ARR, COE, FLPHI, and COEFREL subroutines. FLPHI calls 
for the function WIGMOD. The meaning of the parameters of the OSBRAC subroutine is explained
in comments at its beginning. 
-------------------------------------------------------------------------------------------------------------------------------------------
While it is clear from the comments in the beginnings of ALLOSBRAC and OSBRAC how to
implement these subroutines, this is also obvious from examples listed in the above mentioned
TESTALLOSBRAC and TESTOSBRAC programs. 

In general, to compile codes that include the present programs one may say e.g., 
f95 main.f90 ... allosbrac.f90 ... or, alternatively, f95 main.f90 ... osbrac.f90 ....










