    program testallosbrac
! Tests of the program that calculates the <N_1'L_1'N_2'L_2'|N_1L_1N_2L_2>_L^\varphi brackets.
    implicit double precision(a-h,o-z)
    dimension fac(0:201),dfac(0:201),defac(0:201),rfac(0:100)
! These arrays enter the routine ARR called below. The first three arrays are used merely to
! compute the exact expression for the bracket below. The array rfac is of no use.    
    allocatable :: brac(:,:,:,:,:,:,:,:),bi(:,:)
! Below the allocation/deallocation actions are done independently for each of the tests
! performed as it would be required in the case when only a single test is being performed. 
    open(13,file='allosoutput')
! The input angular parameters are CO=COS(VARPHI) and SI=SIN(VARPHI). Set, for example,
    si=1/sqrt(2.d0)
    co=-si
! This corresponds to Eq. (13) in the accompanying CPC paper in the equal mass case. The
! corresponding transformation (7) in that paper arises at calculating standard two-body 
! shell-model matrix elements.
! 1. Comparison of the value of the <n1p,L00|n1,l1,n2,l2>_L^\varphi bracket with the result 
! of its analytic calculation. Here l1+l2 should be of the same parity as L.
    l=4
    l1=2
    l2=4
    n1=3
    n2=2
    nq=2*n1+2*n2+l1+l2
    mp=0
    np=l
    n1p=(nq-l)/2
! Computation of the exact bracket.
    nfac=2*nq+1
    CALL ARR(FAC,DFAC,DEFAC,RFAC,NQ)
! The array of binomial coefficients to calculate the Wigner coefficients:                
    allocate (BI(0:NFAC,0:NFAC),STAT=istatus)
    IF (istatus /= 0) STOP "TESTOSBRAC: allocate BI failed"
    call coe(NFAC,FAC,BI)
    nm=n1p-n1-n2
    nfa=nm-2*(nm/2)
    LL3=2*L
    N3=L1+L2-L
! exact bracket:        
    b=co**(2*n1+l1)*si**(2*n2+l2)*(-1)**l*sqrt((2.d0*l1+1)*(2*l2+1))*(-1)**nfa*&
    WIGMOD(l1,l2,l,0,0,bi,nfac)*SQRT(BI(L1+L-L2,2*L1)*BI(N3,2*L2)/&
    ((LL3+1)*BI(N3,L1+L2+L+1)*BI(L,LL3)))&
    *sqrt(defac(n1p)*dfac(n1p+l)/(defac(n1)*defac(n2)*dfac(n1+l1)*dfac(n2+l2)))
! bracket from the code:                
    ndel=4
    nqmax=nq+ndel        
! The result should not depend on ndel, provided that ndel is even, i.e., nq and nqmax are of the
! same parity.      
    NQML=NQMAX-L
    NN=0
    m=(l1+l2-l-nn)/2
    n=(l1-l2+l-nn)/2
    allocate (BRAC(0:L,0:(NQMAX-L)/2,0:(NQMAX-L)/2,0:(NQMAX-L)/2,&
    0:(NQMAX-L)/2,0:L,0:(NQMAX-L)/2,L:L),STAT=istatus)
    IF (istatus /= 0) STOP "TESTOSBRAC: allocate(BRAC) failed"
    CALL ALLOSBRAC(NQMAX,L,L,CO,SI,BRAC) 
    a=brac(np,n1p,mp,n1,n2,n,m,l)
    write(6,*)'1. bracket <n1p L 0 0|n1 l1 n2 l2>_L^varphi' 
    write(6,*)'n1 =',n1,' l1 =',l1,' n2 =',n2,' l2 =',l2,' L   =',l
    write(6,*)'exact bracket',b,' bracket from the code',a
    write(13,*)'1. bracket <n1p L 0 0|n1 l1 n2 l2>_L^varphi'
    write(13,*)'n1 =',n1,' l1 =',l1,' n2 =',n2,' l2 =',l2,' L =',l 
    write(13,*)'exact bracket',b,' bracket from the code',a
    deallocate (BRAC,STAT=istatus)
    
! 2. Test of the symmetry of the brackets. The relation 
! <N1P,L1P,N2P,L2P|N1,L1,N2,L2>_L^\varphi=<N1,L1,N2,L2|N1P,L1P,N2P,L2P>_L^\varphi
! should hold true. This symmetry relation may be rewritten as
! BRAC(NP,N1P,MP,N1,N2,N,M,L)=BRAC1(N,N1,M,N1P,N2P,NP,MP,L) where N and
! M represent the angular momenta L1 and L2 and NP and MP represent L1P and L2P. 
! An example:
    nq=30
    l=9
! nn=1 in the present example.
! calculate: nmax=l-nn.
! nmax=8  in the present example. Any n and np that do not exceed nmax may be employed.
    n=4 
    np=3      
! calculate: nqmld2=(nq-l-nn)/2.
! nqmld2=10 in the present example. Any m, n1, n2, mp, n1p, and n2p may be employed
! provided that m+n1+n2=mp+n1p+n2p=nqmld2.
    m=5
    n1=3
    n2=2
    mp=4
    n1p=3
    n2p=3
    nqmax=32
! Recall that nq=30. Any nqmax of the same parity as nq and such that nqmax.ge.nq  
! may be employed. Then its choice does not influence the results.  
    allocate (BRAC(0:L,0:(NQMAX-L)/2,0:(NQMAX-L)/2,0:(NQMAX-L)/2,&
    0:(NQMAX-L)/2,0:L,0:(NQMAX-L)/2,L:L),STAT=istatus)
    IF (istatus /= 0) STOP "TESTOSBRAC: allocate(BRAC) failed" 
    CALL ALLOSBRAC(NQMAX,L,L,CO,SI,BRAC)  
    write(6,*)'2. Symmetry test, L=',l
    write(6,*)'m =',m,' n =',n,' n1 =',n1,' n2 =',n2
    write(6,*)'mp =',mp,' np =',np,' n1p =',n1p,' n2p =',n2p
    write(6,*)'bracket(mp,np,n1p,n2p;m,n,n1,n2) =',BRAC(np,n1p,mp,n1,n2,n,m,l)
    write(6,*)'bracket(m,n,n1,n2;mp,np,n1p,n2p) =',BRAC(n,n1,m,n1p,n2p,np,mp,l)
    write(13,*)'2. Symmetry test, L=',l
    write(13,*)'m =',m,' n =',n,' n1 =',n1,' n2 =',n2
    write(13,*)'mp =',mp,' np =',np,' n1p =',n1p,' n2p =',n2p
    write(13,*)'bracket(mp,np,n1p,n2p;m,n,n1,n2) =',BRAC(np,n1p,mp,n1,n2,n,m,l)
    write(13,*)'bracket(m,n,n1,n2;mp,np,n1p,n2p) =',BRAC(n,n1,m,n1p,n2p,np,mp,l)  
    deallocate (BRAC,STAT=istatus)
! 3. Test of the relation 
! \sum_{i,i',L}\sum_j<j|i><j|i'>=\sum_L N_0(L).
! Here i=(n_1,l_1,n_2,l_2), i'=(n_3,l_3,n_4,l_4), and j=(n_1p,l_1p,n_2p,l_2p). The l.h.s. sum 
! runs over all i and i' values such that nq(i)=nq(i')=nq. 
! In the r.h.s. sum, N_0(L) is the number of the (n_1,l_1,n_2,l_2) states pertaining to
! given nq and L values. The sum runs over all the L values compatible with a given nq value
! and thus represents the total number of states with this nq.
    nq=14
    ndel=4
    nqmax=nq+ndel    
! The result should not depend on ndel, i.e., on nqmax, provided that ndel is even.         
    numb=0
    s=0.d0
    LMIN=0
    LMAX=NQ   
    allocate (BRAC(0:LMAX,0:(NQMAX-LMIN)/2,0:(NQMAX-LMIN)/2,0:(NQMAX-LMIN)/2,&
    0:(NQMAX-LMIN)/2,0:LMAX,0:(NQMAX-LMIN)/2,LMIN:LMAX),STAT=istatus)
    IF (istatus /= 0) STOP "TESTOSBRAC: allocate(BRAC) failed"
    CALL ALLOSBRAC(NQMAX,LMIN,LMAX,CO,SI,BRAC)                   
    do L=0,nq
        nqml=nq-l
        nqmld2=nqml/2  
        nn=nqml-2*nqmld2          
        mmax=(nq-(l+nn))/2
        nmax=l-nn
        nst=(nmax+1)*(mmax+1)*(mmax+2)/2
        do na=0,nmax
            do ma=0,mmax
! IN THE ABOVE NOTATION NA=NA(L1,L2), MA=MA(L1,L2).
                n12=mmax-ma               
                do nb=0,nmax
                    do mb=0,mmax
! IN THE ABOVE NOTATION NB=NB(L3,L4), MB=MB(L4,L4).
                        n34=mmax-mb               
                        ds=0.d0
                        do n2=0,n12
                            n1=n12-n2
                            do n4=0,n34
                                n3=n34-n4
                                DO MP=0,mmax
                                    do n1p=0,mmax-mp
                                        DO NP=0,nmax
                                            DS=DS+BRAC(np,n1p,mp,n1,n2,na,ma,l)&
                                            *BRAC(np,n1p,mp,n3,n4,nb,mb,l)
                                        enddo ! NP
                                    ENDDO ! n1p
                                ENDDO ! MP
                            enddo ! n4
                        enddo ! n2
                        s=s+ds
                    enddo ! MB
                enddo ! NB
            enddo ! MA
        enddo    ! NA
        numb=numb+nst 
    ENDDO !l 
    deallocate (BRAC,STAT=istatus)   
    write(6,*)'3. Test of the relation sum_{i,ip,L}sum_j<j|i><j|ip> = the number of states.'
    write(6,*)'See the comments in the text of the program and in more detail in the' 
    write(6,*)'accompanying CPC paper for further explanations.'  
    write(6,*)'nqmax =',nqmax,' nq =',nq
    write(6,*)'sum_{i1,i2,L}sum_j<j|i1><j|i2> =',s,' exact value (number of states) =', numb
    write(13,*)'3. Test of the relation sum_{i,ip,L}sum_j<j|i><j|ip> = the number of states.'
    write(13,*)'See the comments in the text of the program and in more detail in the' 
    write(13,*)'accompanying CPC paper for further explanations.'  
    write(13,*)'nqmax =',nqmax,' nq =',nq
   write(13,*)'sum_{i1,i2,L}sum_j<j|i1><j|i2> =',s,' exact value (number of states)=',numb         
! 4. Test of the relation \sum_{i,i',L}|\sum_j<j|i><j|i'>-\delta_{i,i'}|=0.
! Here i=(n_1,l_1,n_2,l_2), i'=(n_1',l_1',n_2',l_2'), and j=(n_1'',l_1'',n_2'',l_2''). The outer
! sum runs over all i and i' values such that nq(i)=nq(i') \le nqmax.  
! Also one has nq(i)=nq(i')=nq(j). 
    nqmax=12
    lmin=0
    lmax=nqmax
    nqml=nqmax-lmin
    nqmld2=nqml/2        
    allocate (BRAC(0:LMAX,0:nqmld2,0:nqmld2,0:nqmld2,0:nqmld2,0:LMAX,& 
    0:nqmld2,LMIN:LMAX),STAT=istatus)
    IF (istatus /= 0) STOP "TESTOSBRAC: allocate(BRAC) failed"
    CALL ALLOSBRAC(NQMAX,LMIN,LMAX,CO,SI,BRAC) 
! When nqmax is sufficiently high, this calculation is to be performed in somewhat another way
! because of the memory restrictions. In such cases one may use ALLOSBRAC at lmin=lmax=l.
! The BRAC array is to be allocated and deallocated inside the loop over l below, and
! ALLOSBRAC is to be called inside this loop.   
    s=0.d0
! The quantity s represents the expression \sum_{i,i',L}|\sum_j<j|i><j|i'>-\delta_{i,i'}| we are
! calculating.           
    do l=0,nqmax
    nqml=nqmax-l
    nqmld2=nqml/2
    nn=nqml-2*nqmld2
    nmax=l-nn
    lpnn=l+nn
    do nq=lpnn,nqmax,2
        mmax=(nq-lpnn)/2                       
            do ma=0,mmax
! The equality mmax=(nq-lpnn)/2 follows from the definition of m-type variables.             
                n12=mmax-ma
! The quantity n12 designates n1+n2 and here it is taken into account that
! n1+n2+ma=mmax.                
                do na=0,nmax
! In the case of odd nqmax and l equal to zero, one has nmax=-1 and no contribution arises as it
! should be. 
                    do n1=0,n12                
                        n2=n12-n1                           
                        do mb=0,mmax
                            n34=mmax-mb
                            do nb=0,nmax
                                do n3=0,n34
                                    n4=n34-n3
                                    ds=0.d0 
! The quantity ds represents the contribution of \sum_j|<j|i>_L<j|i'>_L-\delta_{i,i'}| to the net
! result.                                     
                                    DO MP=0,MMAX
                                        do n1p=0,mmax-mp
                                            DO NP=0,NMAX
                                            DS=DS+BRAC(np,n1p,mp,n1,n2,na,ma,l)&
                                            *BRAC(np,n1p,mp,n3,n4,nb,mb,l)
                                            enddo ! NP
                                        ENDDO ! n1p
                                    ENDDO ! MP
                                    if(ma.eq.mb.and.na.eq.nb.and.n1.eq.n3.and.n2.eq.n4)then
                                        ds=abs(ds-1.d0)
                                    else
                                        ds=abs(ds)
                                    endif
                                    s=s+ds
                                enddo ! n3
                            enddo ! nb
                        enddo ! mb
                    enddo ! n1
                enddo ! na
            enddo ! ma
        enddo ! nq
    ENDDO ! l
    deallocate (BRAC,STAT=istatus)
    write(6,*)'4. Test of the relation sum_{i,ip,L}sum_j|<j|i>_L<j|ip>_L-delta_{i,ip}| = 0.'
    write(6,*)'For further explanations see the comments in the text of the program'
    write(6,*)'and, in more detail, in the accompanying CPC paper.'      
    write(6,*)'nqmax =',nqmax
    write(6,*)'sum_{i1,i2,L}|sum_j<j|i1><j|i2>-delta_{i1,i2}| =',s
    write(13,*)'4. Test of the relation sum_{i,ip,L}sum_j|<j|i>_L<j|ip>_L-delta_{i,ip}| = 0.'
    write(13,*)'For further explanations see the comments in the text of the program'
    write(13,*)'and, in more detail, in the accompanying CPC paper.'       
    write(13,*)'nqmax =',nqmax
    write(13,*)'sum_{i1,i2,L}|sum_j<j|i1><j|i2>-delta_{i1,i2}| =',s
    end
