    program testosbrac
! Tests of the program that calculates the <N_1'L_1'N_2'L_2'|N_1L_1N_2L_2>_L^\varphi brackets.
    implicit double precision(a-h,o-z)
    logical firstcall
    dimension fac(0:169),dfac(0:84),defac(0:42),rfac(0:84),bi(0:169,0:169)
! These arrays enter the routine ARR called below. The first three arrays are used merely to
! compute the exact expression for the bracket below. The array rfac is of no use. 
    allocatable :: brac(:,:,:,:,:),brac1(:,:,:,:,:)
    open(13,file='osoutput')
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
    mp=0
    np=l
    nq=2*n1+2*n2+l1+l2 
    n1p=(nq-l)/2
! Computation of the exact bracket.
    nfac=2*nq+1 
    CALL ARR(FAC,DFAC,DEFAC,RFAC,NQ)
! The array of binomial coefficients to calculate the Wigner coefficients:                
    call coe(NFAC,FAC,BI)      
    nm=n1p-n1-n2
    nfa=nm-2*(nm/2)
    LL3=2*L
    N3=L1+L2-L
! exact bracket:        
    b=co**(2*n1+l1)*si**(2*n2+l2)*(-1)**l*sqrt((2.d0*l1+1)*(2*l2+1))*(-1)**nfa*&
    WIGMOD(l1,l2,l,0,0,bi)*SQRT(BI(L1+L-L2,2*L1)*BI(N3,2*L2)/&
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
    allocate (BRAC(0:L,0:NQML/2,0:NQML/2,0:NQML/2,0:NQML/2),STAT=istatus)
    IF (istatus /= 0) STOP "TESTOSBRAC: allocate(BRAC) failed"
    FIRSTCALL=.TRUE.
    CALL OSBRAC(N,M,L,NQMAX,CO,SI,FIRSTCALL,BRAC)
    a=brac(np,n1p,mp,n1,n2)
    write(6,*)'1. bracket <n1p,L00|n1l1n2l2>_L^phi'
    write(6,*)'n1 =',n1,' l1 =',l1,' n2 =',n2,' l2 =',l2,' L =',l 
    write(13,*)'1. bracket <n1p,L00|n1l1n2l2>_L^phi'
    write(13,*)'n1 =',n1,' l1 =',l1,' n2 =',n2,' l2 =',l2,' L =',l 
    write(6,*)'exact bracket',b,' bracket from the code',a
    write(13,*)'exact bracket',b,' bracket from the code',a
    deallocate (BRAC,STAT=istatus)
! 2. Test of the symmetry of the brackets. The relation 
! <N1P,L1P,N2P,L2P|N1,L1,N2,L2>_L^\varphi=<N1,L1,N2,L2|N1P,L1P,N2P,L2P>_L^\varphi
! should hold true. Let the bracket in the left-hand side of this relation is contained in the BRAC
! array and that in its right-hand side in the BRAC1 array. Then the above symmetry relation may
! be rewritten as BRAC(NP,N1P,MP,N1,N2)=BRAC1(N,N1,M,N1P,N2P) where N and M represent
! the angular momenta L1 and L2 and NP and MP represent L1P and L2P.   
! An example:
    nq=30
    l=9
! nn=1 at this choice.
! calculate: nmax=l-nn.
! nmax=8  in the present example. Any n and np values 
! that do not exceed nmax may be employed.
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
    nqmax=36
! Recall that nq=30. Any nqmax of the same parity as nq and such that nqmax.ge.nq  
! may be employed. Then its choice does not influence the results.  
    nqml=nqmax-l 
    allocate (BRAC(0:L,0:NQML/2,0:NQML/2,0:NQML/2,0:NQML/2),STAT=istatus)
    IF (istatus /= 0) STOP "TESTOSBRAC: allocate(BRAC) failed"
    CALL OSBRAC(N,M,L,NQMAX,CO,SI,FIRSTCALL,BRAC)
    allocate (BRAC1(0:L,0:NQML/2,0:NQML/2,0:NQML/2,0:NQML/2),STAT=istatus)
    IF (istatus /= 0) STOP "TESTOSBRAC: allocate(BRAC1) failed"
    CALL OSBRAC(NP,MP,L,NQMAX,CO,SI,FIRSTCALL,BRAC1)
    write(6,*)'2. Symmetry test, L =',l
    write(6,*)'m =',m,' n =',n,' n1=',n1,' n2=',n2,' mp =',mp,' np =',np,' n1p =',n1p,' n2p =',n2p
    write(6,*)'bracket(mp,np,n1p,n2p;m,n,n1,n2) =', BRAC(np,n1p,mp,n1,n2)
    write(6,*)'bracket(m,n,n1,n2;mp,np,n1p,n2p) =', BRAC1(n,n1,m,n1p,n2p)
    write(13,*)'2. Symmetry test, L=',l
    write(13,*)'m =',m,' n =',n,' n1=',n1,' n2=',n2,' mp =',mp,' np =',np,' n1p =',n1p,' n2p =',n2p
    write(13,*)'bracket(mp,np,n1p,n2p;m,n,n1,n2) =', BRAC(np,n1p,mp,n1,n2)
    write(13,*)'bracket(m,n,n1,n2;mp,np,n1p,n2p) =', BRAC1(n,n1,m,n1p,n2p)  
    deallocate (BRAC1,STAT=istatus)
    deallocate (BRAC,STAT=istatus)         
! 3. Test of the relation 
! \sum_{i,i',L}\sum_j<j|i>_L<j|i'>_L=\sum_L N_0(L).
! Here i=(n_1,l_1,n_2,l_2), i'=(n_3,l_3,n_4,l_4), and j=(n_1p,l_1p,n_2p,l_2p). The l.h.s. sum 
! runs over all i and i' values such that nq(i)=nq(i')=nq. 
! In the r.h.s. sum, N_0(L) is the number of the (n_1,l_1,n_2,l_2) states pertaining to
! given nq and L values. The sum runs over all the L values compatible with a given nq value
! and thus represents the total number of states with this nq.
    nq=14
    ndel=6
    nqmax=nq+ndel
! The result should not depend on ndel, i.e., on nqmax, provided that ndel is even.         
    numb=0
    s=0.d0                   
    do l=0,nq
        FIRSTCALL=.true.
        nqml=nqmax-l
        nqmld2=nqml/2    
        nn=nqml-2*nqmld2
        mmax=(nq-(l+nn))/2
        nmax=l-nn
        nst=(nmax+1)*(mmax+1)*(mmax+2)/2
        allocate (BRAC(0:L,0:NQMLD2,0:NQMLD2,0:NQMLD2,0:NQMLD2),STAT=istatus)
        IF (istatus /= 0) STOP "TESTOSBRAC: allocate(BRAC) failed"
        allocate (BRAC1(0:L,0:NQMLD2,0:NQMLD2,0:NQMLD2,0:NQMLD2),STAT=istatus)
        IF (istatus /= 0) STOP "TESTOSBRAC: allocate(BRAC1) failed"
! The allocation/deallocation of BRAC and BRAC1, at nqmld2-->nqmax/2, might also be done
! outside the loop.        
        do ma=0,mmax
            n12=mmax-ma 
            do na=0,nmax 
! IN THE ABOVE NOTATION NA=NA(L1,L2) AND MA=MA(L1,L2).                      
                CALL OSBRAC(NA,MA,L,NQMAX,CO,SI,FIRSTCALL,BRAC)
! The order of summations below aims to avoid unnecessary calls of osbrac.                 
                 do mb=0,mmax
                    n34=mmax-mb 
                    do nb=0,nmax 
! IN THE ABOVE NOTATION NB=NB(L3,L4) AND MB=MB(L3,L4).                                  
                        CALL OSBRAC(NB,MB,L,NQMAX,CO,SI,FIRSTCALL,BRAC1)
                        ds=0.d0
                        do n2=0,n12
                            n1=n12-n2
                            do n4=0,n34
                                n3=n34-n4
                                DO MP=0,mmax
                                    do n1p=0,mmax-mp
                                        DO NP=0,nmax
                                            DS=DS+BRAC(NP,n1p,MP,n1,n2)*BRAC1(NP,n1p,MP,n3,n4)
                                        enddo ! NP
                                    ENDDO ! n1p
                                ENDDO ! MP
                            enddo ! n4
                        enddo ! n2
                        s=s+ds
                    enddo ! nb
                    FIRSTCALL=.false.
! If the nb loop is nested inside the mb loop the FIRSTCALL=.false. command is more efficient.
! In practice, usually the gain in the running time is relatively not large.                     
                enddo ! mb
            enddo ! ma
        enddo    ! na
        numb=numb+nst
        deallocate (BRAC,STAT=istatus)
        deallocate (BRAC1,STAT=istatus)
    ENDDO ! l
    write(6,*)'3. Test of the relation sum_{i,ip,L,j}<j|i>_L<j|ip>_L = the number of states.'
    write(6,*)'See the comments in the text of the program and in more detail in the' 
    write(6,*)'accompanying CPC paper for further explanations.'  
    write(6,*)'nqmax =',nqmax,' nq =',nq
    write(6,*)'sum_{i,ip,L,j}<j|i>_L<j|ip>_L  =',s,' exact value (number of states) =',numb
    write(13,*)'3. Test of the relation sum_{i,ip,L}sum_j<j|i><j|ip> = the number of states.'
    write(13,*)'See the comments in the text of the program and in more detail in the' 
    write(13,*)'accompanying CPC paper for further explanations.'  
    write(13,*)'nqmax =',nqmax,' nq =',nq         
    write(13,*)'sum_{i,ip,L,j}<j|i>_L<j|ip>_L  =',s,' exact value (number of states) =',numb                      
! 4. Test of the relation \sum_{i,i',L}\sum_j|<j|i>_L<j|i'>_L-\delta_{i,i'}|=0.
! Here i=(n_1,l_1,n_2,l_2), i'=(n_1',l_1',n_2',l_2'), and j=(n_1'',l_1'',n_2'',l_2''). The outer
! sum runs over all i and i' values such that nq(i)=nq(i') \le nqmax.  
! Also one has nq(i)=nq(i')=nq(j).        
    nqmax=12
    s=0.d0 
! The quantity s represents the expression \sum_{i,i',L}|\sum_j<j|i><j|i'>-\delta_{i,i'}| we are
! calculating.                
    do l=0,nqmax
        FIRSTCALL=.true.   
        nqml=nqmax-l
        nqmld2=nqml/2    
        nn=nqml-2*nqmld2
        nmax=l-nn
        lpnn=l+nn
        allocate (BRAC(0:L,0:NQMLD2,0:NQMLD2,0:NQMLD2,0:NQMLD2),STAT=istatus)
        IF (istatus /= 0) STOP "TESTOSBRAC: allocate(BRAC) failed"
        allocate (BRAC1(0:L,0:NQMLD2,0:NQMLD2,0:NQMLD2,0:NQMLD2),STAT=istatus)
        IF (istatus /= 0) STOP "TESTOSBRAC: allocate(BRAC1) failed"
! The allocation/deallocation of BRAC and BRAC1, at nqmld2-->nqmax/2, might also be done
! outside the loop.          
        do ma=0,nqmld2 
! The above upper limit of summation corresponds to the definition of m-type
! variables.                          
            do na=0,nmax            
! In the case of odd nqmax and l equal to zero, one has nmax=-1 and no contribution arises as it
! should be.     
                CALL OSBRAC(NA,MA,L,NQMAX,CO,SI,FIRSTCALL,BRAC)
! The order of summations below aims to avoid unnecessary calls of osbrac.                 
                do mb=0,nqmld2
                    do nb=0,nmax                    
                        CALL OSBRAC(NB,MB,L,NQMAX,CO,SI,FIRSTCALL,BRAC1)                        
                        do nq=l+nn,nqmax,2
                            nqcmld2=(nq-l-nn)/2
                            n12=nqcmld2-ma
                            n34=nqcmld2-mb
! We consider nq to be the number of quanta pertaining to states entering the brackets in our
! sum. Then one has: ma+n1+n2=mb+n3+n4=mp+n1p+n2p=(nq-l-nn)/2 that is
! nqcmld2.                           
                            do n1=0,n12
                                n2=n12-n1
                                do n3=0,n34
                                    n4=n34-n3                                     
                                    ds=0.d0 
! The quantity ds represents the contribution of \sum_j|<j|i>_L<j|i'>_L-\delta_{i,i'}| to the net
! result.                                              
                                    DO MP=0,nqcmld2
                                        do n1p=0,nqcmld2-mp
                                            DO NP=0,NMAX
                                                DS=DS+BRAC(NP,n1p,MP,n1,n2)*BRAC1(NP,n1p,MP,n3,n4)
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
                            enddo ! n1
                        enddo ! nq
                    enddo ! nb
                    FIRSTCALL=.false.
! If the nb loop is nested inside the mb loop the FIRSTCALL=.false. command is more efficient.
! In practice, usually the gain in the running time is relatively not large. 
                enddo ! mb
            enddo ! ma
        enddo ! na
        deallocate (BRAC,STAT=istatus)
        deallocate (BRAC1,STAT=istatus)
    ENDDO ! l
    write(6,*)'4. Test of the relation sum_{i,ip,L}sum_j|<j|i>_L<j|ip>_L-delta_{i,ip}| = 0.'
    write(6,*)'For further explanations see the comments in the text of the program'
    write(6,*)'and, in more detail, in the accompanying CPC paper.'  
    write(6,*)'nqmax =',nqmax
    write(6,*)'sum_{i1,i2,L}|sum_j<j|i1>_L<j|i2>_L-delta_{i1,i2}| =',s
    write(13,*)'4. Test of the relation sum_{i,ip,L}sum_j|<j|i>_L<j|ip>_L-delta_{i,ip}| = 0.'
    write(13,*)'For further explanations see the comments in the text of the program'
    write(13,*)'and, in more detail, in the accompanying CPC paper.'  
    write(13,*)'nqmax =',nqmax
    write(13,*)'sum_{i1,i2,L}|sum_j<j|i1>_L<j|i2>_L-delta_{i1,i2}| =',s
    end
