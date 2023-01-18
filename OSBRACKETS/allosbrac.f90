    SUBROUTINE ALLOSBRAC(NQMAX,LMIN,LMAX,CO,SI,BRAC)
    
! COMMENTS ON USE:
!---------------------------------------------------------------------------------------------------------------------------------------
! THE OSCILLATOR BRACKETS <N1P,L1P,N2P,L2P|N1,L1,N2,L2>_L^\VARPHI ARE 
! CALCULATED  HERE. THIS IS DONE WITH THE HELP OF EQS. (18), (20)-TYPE, AND (22) IN 
! THE ACCOMPANYING CPC PAPER. THE QUANTITIES L1, L2, L1P, AND L2P IN THE ABOVE
! DEFINITION OF THE BRACKETS ARE THE PARTIAL ANGULAR MOMENTA, L IS THE TOTAL 
! ANGULAR MOMENTUM, AND N1, N2, N1P, N2P ARE  THE RADIAL QUANTUM NUMBERS.
! IN THE NOTATION LIKE L1P, ETC., "P" SYMBOLIZES "PRIMED" HERE AND BELOW.

! THE SUBROUTINE RETURNS THE ARRAY OF ALL THE BRACKETS SUCH THAT
! L1+L2+2*(N1+N2).LE.NQMAX, LMIN \LE L \LE LMAX, AND THE BRACKETS PERTAIN TO STATES
! OF THE SAME PARITY (-1)^(L1+L2) WHICH IS THE PARITY OF NQMAX. 
 
! THE L1, L2, L1P, AND L2P ORBITAL MOMENTA ARE EXPRESSED IN THE SUBROUTINE IN 
! TERMS OF THE M, N, MP, AND NP VARIABLES DEFINED AS FOLLOWS
! M=(L1+L2-L-NN)/2, N=(L1-L2+L-NN)/2, MP=(L1P+L2P-L-NN)/2, NP=(L1P-L2P+L-NN)/2
! WHERE NN EQUALS 0 OR 1 WHEN, RESPECTIVELY, NQMAX-L IS EVEN OR ODD. ONE THEN HAS
! L1 = M+N+NN,  L2 = M-N+L, L1P = MP+NP+NN,  L2P = MP-NP+L.
! WHEN L1, L2, L1P AND L2P TAKE ALL THE VALUES ALLOWED AT GIVEN L AND GIVEN PARITY 
! THE N AND NP VARIABLES TAKE ALL THE INTEGER VALUES FROM ZERO UP TO L-NN AND 
! THE M AND MP VARIABLES TAKE ALL THE INTEGER VALUES FROM ZERO UP TO (NQMAX-L-NN)/2.
! ONE ALSO HAS NQ=2*(M+N1+N2)+L+NN=2*(MP+N1P+N2P)+L+NN.  

! ALL THE PARAMETERS OF THE SUBROUTINE BUT BRAC ARE INPUT ONES. THE MEANING
! OF THE NQMAX, LMIN, AND LMAX PARAMETERS IS SEEN FROM THE ABOVE DESCRIPTION.

! THE BRAC PARAMETER DESIGNATES THE ARRAY OF OUTPUT BRACKETS. IT IS OF THE FORM
! BRAC(NP,N1P,MP,N1,N2,N,M,L) . (AS SAID ABOVE, L1=L1(N,M), L2=L2(N,M), L1P=L1P(NP.MP),
! AND L2P=L2P(NP,MP). THE QUANTITY N2P IS DETERMINED BY THE EQUALITY 
! MP+N1P+N2P=M+N1+N2.) THE ORDER OF THE ARGUMENTS OF BRAC CORRESPONDS TO 
! NESTING OF THE LOOPS AT ITS COMPUTATION.

! THE ROUTINE PARAMETERS CO AND SI ARE AS FOLLOWS, CO = COS(PHI) AND SI = SIN(PHI). 
! THESE QUANTITIES DEFINE THE PSEUDO(!!!)ORTHOGONAL TRANSFORMATION 
! XI1P = CO*XI1+SI*XI2, XI2P = SI*XI1-CO*XI2. 

! (BRACKETS PERTAINING TO THE CASE OF THE ORTHOGONAL TRANSFORMATION 
! XI1P = CO*XI1+SI*XI2, XI2P = -SI*XI1+CO*XI2  ARE SIMPLY EXPRESSED IN TERMS OF 
! THOSE CALCULATED IN THE PRESENT PROGRAM, SEE THE ACCOMPANYING CPC PAPER.

! THE MEANING OF THE ARRAYS, OTHER THAN BRAC, ENTERING THE DIMENSION LIST IS AS
! FOLLOWS. B IS A SUBSIDIARY ARRAY TO PERFORM THE RECURSION. 

! FAC(I)=I!, DFAC(I)=(2I+1)!!, AND DEFAC(I)=(2I)!!. RFAC(I)=SQRT((2I)!)/I! 
! THESE ARRAYS ARE PRODUCED BY THE "ARR" ROUTINE WHICH IS CALLED FROM THE PRESENT
! ROUTINE AND WHICH IS CONTAINED IN THE PRESENT FILE. THE SET UPPER BOUNDS OF THESE
! ARRAYS ARE SUFFICIENT FOR THE COMPUTATION. THESE ARRAYS ARE USED IN THE PRESENT
! ROUTINE AND IN THE "FLPHI" ROUTINE.

! THE A ARRAY IS A(N,L)=(-1)**N/SQRT(DEFAC(N)*DFAC(N+L)). IT APPEARS BOTH IN THE N1=N2=0
! BRACKETS AND IN THE RELATION BETWEEN THE < | > AND [ | ] TYPE BRACKETS.

! THE BI ARRAY IS BI(M,N)=FAC(N)/[FAC(M)*FAC(N-M)]. IT ENTERS THE 3J SYMBOLS. IT IS USED
! IN THE "FLPHI" ROUTINE AND IN THE FUNCTION WIGMOD.

! THE FL(NP,MP,N) ARRAY REPRESENTS THE QUANTITY 
! [(2*L1P+1)*(2*L2P+1)]^{1/2}*F_L^VARPHI WHERE F_L^VARPHI IS GIVEN BY EQ. (23) IN THE
! ACCOMPANYING CPC PAPER. THIS ARRAY IS PRODUCED IN ADVANCE BY THE
! "FLPHI" ROUTINE WHICH IS CALLED FROM THE PRESENT ROUTINE AND WHICH IS
! CONTAINED IN THE PRESENT FILE. THE VARIABLES MP, NP, AND N ARE DEFINED ABOVE ALONG
! WITH THEIR UPPER BOUNDS. 

! THE PSIP(P,Q) AND PSIM(P,Q) ARRAYS REPRESENT THE QUANTITIES (4) IN THE ACCOMPANYING
! CPC PAPER. THEY ARE PRODUCED IN ADVANCE BY THE "COEFREC" ROUTINE
! WHICH IS CALLED FROM THE PRESENT ROUTINE AND WHICH IS CONTAINED IN THE PRESENT
! FILE. THEIR ARGUMENTS P AND Q TAKE THE VALUES L1P, L1P+/-1 AND L2P, L2P+/-1,
! RESPECTIVELY. THE BOUNDS OF THE PSIP AND PSIM ARRAYS ARE SUCH THAT ALL THE P AND Q
! VALUES REQUIRED TO PERFORM THE RECURSION ARE PROVIDED.

! THE MENTIONED "FLPHI" ROUTINE USES THE FUNCTION WIGMOD WHICH IS ALSO CONTAINED
! IN THE PRESENT FILE.
!----------------------------------------------------------------------------------------------------------------------------------------
    DOUBLE PRECISION BRAC,FL,PSIP,PSIM,A,FAC,DFAC,DEFAC,RFAC,BI,CO,SI,T,CO2,SI2,SC,FA,&
    PLFACT,D2L12P,TN,PA,S,FA1,FA2 
    DIMENSION& 
    BRAC(0:LMAX,0:(NQMAX-LMIN)/2,0:(NQMAX-LMIN)/2,0:(NQMAX-LMIN)/2,&
    0:(NQMAX-LMIN)/2,0:LMAX,0:(NQMAX-LMIN)/2,LMIN:LMAX),&
! THE ARRAY IS OF THE FORM BRAC(NP,N1P,MP,N1,N2,N,M,L)      
    FL(0:LMAX,0:(NQMAX-LMIN)/2,0:LMAX),&
    PSIP((NQMAX+LMAX)/2+1,(NQMAX+LMAX)/2+1),&
    PSIM((NQMAX+LMAX)/2+1,(NQMAX+LMAX)/2+1),A(0:NQMAX/2+1,0:NQMAX),&
    FAC(0:2*NQMAX+1),DFAC(0:NQMAX+1),DEFAC(0:NQMAX/2+1),RFAC(0:NQMAX),&
    BI(0:2*NQMAX+1,0:2*NQMAX+1) 
    IF (NQMAX.GT.84) THEN
        WRITE(6,*)'IN THE R(*8) COMPUTATION NQMAX SHOULD NOT EXCEED 84' 
        STOP
    ENDIF   
    IF (CO.EQ.0.D0.OR.SI.EQ.0.D0) THEN
        WRITE(6,*)'THE PROGRAM IS OF NO USE AT ZERO COS(PHI) OR SIN(PHI) VALUES'
        WRITE(6,*)'COS(PHI)=',CO,' SIN(PHI)=',SI
        STOP
    ENDIF
    IF (NQMAX.LT.LMAX) THEN
        WRITE(6,*)'NQMAX=',NQMAX,' L=',LMAX
        WRITE(6,*)'L SHOULD NOT EXCEED NQMAX'
        STOP
    ENDIF
    T=SI/CO
    CO2=CO**2 
    SI2=SI**2
    SC=SI*CO
    CALL ARR(FAC,DFAC,DEFAC,RFAC,NQMAX)
    CALL COE(2*NQMAX+1,FAC,BI)
!   THE A_{NL} ARRAY:
    DO NI=0,NQMAX/2+1
        NA=NI-2*(NI/2)
        K=1
        IF(NA.EQ.1)K=-1
        DO LI=0,NQMAX-NI
            A(NI,LI)=K/SQRT(DEFAC(NI)*DFAC(NI+LI))
        ENDDO
    ENDDO             
! BRACKETS [N_1'L_1'N_2'L_2'|0L_10L_2]_L^\PHI TO START THE RECURSION, EQ. (22) IN  
! THE ACCOMPANYING CPC PAPER. 
! THESE BRACKETS ARE REPRESENTED AS BRAC(NP,N1P,MP,0,0,N,M,L). 
! SUBSIDIARY QUANTITIES: 
    DO L=LMIN,LMAX        
        CALL FLPHI(NQMAX,LMAX,LMIN,L,T,BI,RFAC,FL)
! THIS SUBROUTINE PRODUCES THE ARRAY FL USED BELOW.    
        NQML=NQMAX-L
        NQMLD2=NQML/2
        NN=NQML-2*(NQMLD2)
        CALL COEFREC((NQMAX+LMAX)/2+1,L,NN,NQMAX,PSIP,PSIM) 
        NMAX=L-NN 
        MMAX=(NQML-NN)/2
        DO M=0,MMAX
            DO N=0,NMAX     
                L1=M+N+NN 
                L2=M-N+L
                L1L2=L1+L2
                L1L2ML=L1L2-L    
                FA=SQRT((2.D0*L1+1)*(2*L2+1))*CO**L1L2
                PLFACT=SQRT(FAC(L1L2+L+1)*FAC(L1L2ML))
! COMPUTATION OF THE N1=N2=0 BRACKETS. THEY ARE REPRESENTED AS 
! BRAC(NP,N1P,MP,0,0,N,M,L). 
                MPMAX=(L1L2ML-NN)/2
                DO MP=0,MPMAX
                    L12P=2*MP+L+NN
! THIS IS BY DEFINITION, L12P=L1P+L2P
                    D2L12P=1.D0/2.D0**L12P
                    N12P=MPMAX-MP
! N12P=N1P+N2P. FOR THE BRACKETS WE COMPUTE NOW NQMAX=L1L2. 
! THEN N12P=(L1L2-L12P)/2. SINCE MPMAX=(L1L2-L-NN)/2 AND MP=(L12P-L-NN)/2 ONE GETS THIS.
                    DO N1P=0,N12P
                        N1PA=N1P-2*(N1P/2)
                        IF (N1PA.EQ.0) THEN
                            K=1
                        ELSE
                            K=-1
                        ENDIF
! K=(-1)**N1P
                        N2P=N12P-N1P
                        TN=T**N12P*K
                        L1T=L1-N12P
                        L2T=L2-N12P
                        DO NP=0,NMAX
                            L1P=MP+NP+NN
                            L2P=L12P-L1P  
! BY DEFINITION ONE ALSO HAS: L2P=L2P(MP,NP)=MP-NP+L
                            PA=A(N1P,L1P)*A(N2P,L2P)
                            BRAC(NP,N1P,MP,0,0,N,M,L)=FL(NP,MP,N)*FA*TN*D2L12P&
                            *PLFACT*PA*PA
                        ENDDO ! NP
                    ENDDO ! N1P
                ENDDO ! MP
! RECURSION TO OBTAIN THE GENERAL FORM BRACKETS 
! [N_1'L_1'N_2'L_2'|N_1L_1N_2L_2]_L^\PHI. 
                N12MAX=(NQMAX-L1L2)/2
                MPMAX0=MPMAX
! THE N1=0, N2-1-->N2 RECURSION, EQ. (20)  IN THE ACCOMPANYING CPC PAPER WITH THE
! MODIFICATION POINTED OUT THERE: 
                DO N2=1,N12MAX
                    MPMAX=MPMAX+1
! THIS MPMAX VALUE IS (NQ-L-NN)/2 WHERE NQ=L1+L2+2*N2.
                    DO MP=0,MPMAX
                        N12P=MPMAX-MP
                        DO N1P=0,N12P
! MP=(L1P+L2P-L-NN)/2. THEREFORE, N12P=MPMAX-MP=(NQ-L1P-L2P)/2=N1P+N2P.
                            DO NP=0,NMAX
                                L1P=MP+NP+NN
                                L2P=MP-NP+L  
                                S=0.D0 
! BELOW THE RESTRICTIONS ARE IMPOSED ON THE BRAC ARRAY ENTERING THE RIGHT-HAND
! SIDE OF THE RECURRENCE FORMULAE. THIS ARRAY IS OF THE FORM BRAC(K1,K2,K3,...) WHERE
! K1=K1(NP), K2=N2(N1P), AND K3=K3(MP). NAMELY, K1=NP, OR NP+1, OR NP-1; 
! K2=N1P OR N1P-1; K3=MP, OR MP-1, OR MP+1. (THE COMBINATION K2=N1P AND K3=MP+1
! DOES NOT ARISE IN THE RECURRENCE FORMULAE.) THE QUANTITIES K1, K2, AND K3
! REPRESENT, RESPECTIELY, NP, N1P, AND MP VALUES AT THE PRECEDING STAGE OF THE
! RECURSION. THE RESTRICTIONS ENSURE THAT K1, K2, AND K3 RANGE WITHIN THE LIMITS
! PERTAINING TO THE BRAC ARRAY OBTAINED AT THAT PRECEDING STAGE OF THE RECURSION. 

! THUS K1, K2, AND K3 SHOULD BE NON-NEGATIVE. THEREFORE, WHEN K1=NP-1 THE VALUE
! NP=0 IS TO BE EXCLUDED, WHEN K2=N1P-1 THE VALUE N1P=0 IS TO BE EXCLUDED, AND
! WHEN K3=MP-1 THE VALUE MP=0 IS TO BE EXCLUDED. 

! FURTHERMORE, IT SHOULD BE K1 \LE NMAX AND AT THE SAME TIME ONE HAS NP \LE NMAX.
! THEREFORE, WHEN K1=NP+1 THE VALUE NP=NPMAX IS TO BE EXCLUDED.

! IN ADDITION, IT SHOULD BE K2+K3 \LE MPMAX-1 WHILE ONE HAS N1P+MP \LE MPMAX. WHEN
! K2=N1P AND K3=MP-1, OR K2=N1P-1 AND K3=MP-1, OR K2=N1P-1 AND K3=MP ONE
! AUTOMATICALLY HAS N1P+MP \LE MPMAX. THEREFORE, IN THESE CASES THE CONDITION
! K2+K3 \LE MPMAX-1 DOES NOT CREATE ANY RESTRICTIONS. 
! BUT WHEN K2=N1P AND K3=MP, OR K2=N1P-1 AND K3=MP+1 THE CASE N1P+MP=MPMAX
! IS TO BE FORBIDDEN. 

! IN THE RECURRENCE FORMULAE BELOW THE DESCRIBED RESTRICTIONS ARE
! IMPOSED.   
                                IF (MP.NE.0) S=BRAC(NP,N1P,MP-1,0,N2-1,N,M,L)*PSIP(L1P,L2P)
                                IF (N1P.NE.0.AND.N1P.NE.N12P) S=S+&
                                BRAC(NP,N1P-1,MP+1,0,N2-1,N,M,L)*PSIP(L1P+1,L2P+1)
! RECALL THAT N12P=MPMAX-MP.                                
                                IF (NP.NE.NMAX.AND.N1P.NE.0) S=S-&                               
                                BRAC(NP+1,N1P-1,MP,0,N2-1,N,M,L)*PSIM(L1P+1,L2P)                                
                                IF (NP.NE.0.AND.N1P.NE.N12P) S=S-&
                                BRAC(NP-1,N1P,MP,0,N2-1,N,M,L)*PSIM(L1P,L2P+1)                                
                                S=S*SC                                
                                IF (N1P.NE.0) S=S+BRAC(NP,N1P-1,MP,0,N2-1,N,M,L)*SI2
                                IF (N1P.NE.N12P) S=S+BRAC(NP,N1P,MP,0,N2-1,N,M,L)*CO2
                                BRAC(NP,N1P,MP,0,N2,N,M,L)=S
                            ENDDO ! NP
                        ENDDO ! N1P
                    ENDDO ! MP
                ENDDO ! N2
! N1-1-->N1 RECURSION, EQ. (20) IN THE ACCOMPANYING CPC PAPER:
                DO N2=0,N12MAX
                    MPMAX=MPMAX0+N2
! THE CURRENT MPMAX VALUE IS (NQ-L-NN)/2=MPMAX0+N2 SINCE NQ=L1+L2+2*N2.
! THE RECURSION:        
                    DO N1=1,N12MAX-N2
                        MPMAX=MPMAX+1
! THE CURRENT MPMAX VALUE IS (NQ-L-NN)/2 AND NQ=L1+L2+2*N2+2*N1.            
                        DO MP=0,MPMAX
                            N12P=MPMAX-MP
                            DO N1P=0,N12P
! MP=(L1P+L2P-L-NN)/2. THEREFORE, N1PMAX=MPMAX-MP=(NQ-L1P-L2P)/2=N1P+N2P.              
                                DO NP=0,NMAX
                                    L1P=MP+NP+NN
                                    L2P=MP-NP+L  
                                    S=0.D0                 
                                    IF (MP.NE.0) S=BRAC(NP,N1P,MP-1,N1-1,N2,N,M,L)*PSIP(L1P,L2P)
                                    IF (N1P.NE.0.AND.N1P.NE.N12P) S=S+&
                                    BRAC(NP,N1P-1,MP+1,N1-1,N2,N,M,L)*PSIP(L1P+1,L2P+1)
                                    IF (NP.NE.NMAX.AND.N1P.NE.0) S=S-&
                                    BRAC(NP+1,N1P-1,MP,N1-1,N2,N,M,L)*PSIM(L1P+1,L2P)
                                    IF (NP.NE.0.AND.N1P.NE.N12P) S=S-&
                                    BRAC(NP-1,N1P,MP,N1-1,N2,N,M,L)*PSIM(L1P,L2P+1)
! THESE RELATIONS ARE THE SAME AS THE CORRESPONDING ONES ABOVE.
                                    S=-S*SC
                                    IF (N1P.NE.0) S=S+BRAC(NP,N1P-1,MP,N1-1,N2,N,M,L)*CO2
                                    IF (N1P.NE.N12P) S=S+BRAC(NP,N1P,MP,N1-1,N2,N,M,L)*SI2
! THESE RELATIONS ARE THE SAME AS THE CORRESPONDING ONES ABOVE.
                                    BRAC(NP,N1P,MP,N1,N2,N,M,L)=S
                                ENDDO ! NP
                            ENDDO ! N1P
                        ENDDO ! MP
                    ENDDO ! N1
                ENDDO ! N2
! RENORMALIZATION OF THE BRACKETS, EQ. (18) IN THE ACCOMPANYING CPC PAPER:
                DO N2=0,N12MAX
                    MPMAX1=MPMAX0+N2
                    DO N1=0,N12MAX-N2
                        MPMAX=MPMAX1+N1
                        FA1=A(N1,L1)*A(N2,L2)
                        DO MP=0,MPMAX
                            N12P=MPMAX-MP
                            DO N1P=0,MPMAX-MP
                                N2P=N12P-N1P
                                DO NP=0,NMAX
                                    L1P=MP+NP+NN
                                    L2P=MP-NP+L                       
                                    FA2=A(N1P,L1P)*A(N2P,L2P)
                                    BRAC(NP,N1P,MP,N1,N2,N,M,L)=BRAC(NP,N1P,MP,N1,N2,N,M,L) &
                                    *FA1/FA2
                                ENDDO ! NP
                            ENDDO ! N1P 
                        ENDDO ! MP
                    ENDDO ! N1
                ENDDO ! N2                      
            ENDDO ! N
        ENDDO ! M
    ENDDO ! L
    RETURN
    END 
 
    SUBROUTINE ARR(FAC,DFAC,DEFAC,RFAC,NQMAX)
! FAC(I), DFAC(I),DEFAC(I), AND RFAC(I) ARE, RESPECTIVELY, THE QUANTITIES
! I!, (2I+1)!!, (2I)!!, AND SQRT((2*I)!)/I!
    DOUBLE PRECISION FAC,DFAC,DEFAC,RFAC
    DIMENSION FAC(0:2*NQMAX+1),DFAC(0:NQMAX+1),DEFAC(0:NQMAX/2+1),RFAC(0:NQMAX)
    FAC(0)=1.D0
    DFAC(0)=1.D0
    DEFAC(0)=1.D0
    RFAC(0)=1.D0
    DO I=1,2*NQMAX+1 
        FAC(I)=FAC(I-1)*I 
    ENDDO            
    DO I=1,NQMAX+1           
        DFAC(I)=DFAC(I-1)*(2*I+1)
    ENDDO    
    DO I=1,NQMAX/2+1
        DEFAC(I)=DEFAC(I-1)*2*I
    ENDDO
    DO I=1,NQMAX    
        RFAC(I)=RFAC(I-1)*2*SQRT(1-0.5D0/I)    
    ENDDO
    RETURN
    END
    

    SUBROUTINE FLPHI(NQMAX,LMAX,LMIN,L,T,BI,RFAC,FL)
! PROVIDES THE QUANTITY SQRT((2L1P+1)*(2L2P+1))*F_L^\VARPHI, SEE EQ. (23) IN THE 
! ACCOMPANYING PAPER, IN THE FORM OF THE FL ARRAY.
! USES THE FUNCTION WIGMOD. 
! THIS FLPHI DIFFERS FROM THAT IN THE OTHER FILE osbrac.f90.
    DOUBLE PRECISION BI,RFAC,FL,T,T2,SQP,F,WIGMOD
    DIMENSION&
    FL(0:LMAX,0:(NQMAX-LMIN)/2,0:LMAX),BI(0:2*NQMAX+1,0:2*NQMAX+1),RFAC(0:NQMAX)
! THE OUTPUT IS FL(NP,MP,N) WHERE MP=(L1P+L2P-L-NN)/2, L1P+L2P=L1T+L2T,
! NP=(L1P-L2P+L-NN)/2, AND N=(L1-L2+L-NN)/2, L1-L2=L1T-L2T. 
    T2=T*T    
    NQMAL=NQMAX-L
    MPMAX=NQMAL/2
    NN=NQMAL-2*MPMAX
    LMNN=L-NN 
    LL3=2*L
    DO N=0,LMNN
        L1ML2=2*N-LMNN 
        DO MP=0,MPMAX
            L1PL2P=2*MP+L+NN
            DO  NP=0,LMNN
                L1PML2P=2*NP-LMNN            
                L1P=(L1PL2P+L1PML2P)/2
                L2P=(L1PL2P-L1PML2P)/2
                M3=L2P-L1P
                SQP=SQRT((2*L1P+1.D0)*(2*L2P+1))
                L1T=(L1PL2P+L1ML2)/2
                L2T=(L1PL2P-L1ML2)/2
                N3=L1T+L2T-L
                IMIN=ABS(L1T-L1P)
                IMAX=MIN(L1P+L1T,L2P+L2T)
                NAL1=(L1T+L1P-IMIN)/2
                NAL2=(L1T-L1P+IMIN)/2
                NAL3=(L2T-L2P+IMIN)/2
                NAL4=(L2T+L2P-IMIN)/2
                NAL4P=NAL4-2*(NAL4/2)
                F=0.D0
                LY=1
                DO I=IMIN,IMAX,2                          
                    F=F+RFAC(NAL1)*RFAC(NAL2)*RFAC(NAL3)*RFAC(NAL4)*&
                    WIGMOD(L1T,L2T,L,L1P-I,I-L2P,BI,2*NQMAX+1)*T2*LY
                    NAL1=NAL1-1
                    NAL2=NAL2+1
                    NAL3=NAL3+1
                    NAL4=NAL4-1    
                    LY=-LY
                ENDDO
                FL(NP,MP,N)=F*T**IMIN*SQRT((2*L1P+1)*(2*L2P+1)&
                *BI(L1T+L-L2T,2*L1T)*BI(N3,2*L2T)/((LL3+1)*BI(N3,L1T+L2T+L+1)*BI(L+M3,LL3)))
                IF(NAL4P.NE.0)FL(NP,MP,N)=-FL(NP,MP,N)
            ENDDO
        ENDDO
    ENDDO
    RETURN
    END
          
   FUNCTION WIGMOD(L1,L2,L3,M1,M2,BI,NMAX)
! THE 3J SYMBOL IN TERMS OF BINOMIAL COEFFICIENTS WITHOUT THE FACTOR
! SQRT(F) WHERE F IS AS FOLLOWS,
! F=BI(L1+L3-L2,2*L1)*BI(N3,2*L2)/((LL3+1)*BI(N3,L1+L2+L3+1)*BI(L3+M3,LL3)) 
! WITH N3=L1+L2-L3, LL3=2*L3, AND M3=-M1-M2.   
    DOUBLE PRECISION WIGMOD,BI,S
    DIMENSION BI(0:NMAX,0:NMAX)
    M3=-M1-M2
    N3=L1+L2-L3
    LM1=L1-M1
    LP2=L2+M2
    KMIN=MAX(0,L1-L3+M2,L2-L3-M1)
    KMAX=MIN(N3,LM1,LP2)
    S=0.d0
    NPH=1
    DO K=KMIN,KMAX
        S=S+NPH*BI(K,N3)*BI(LM1-K,L1+L3-L2)*BI(LP2-K,L2+L3-L1)
        NPH=-NPH
    ENDDO 
    WIGMOD=S/SQRT(BI(LM1,2*L1)*BI(LP2,2*L2))
    NY=KMIN+L1-L2-M3
    NYP=NY-2*(NY/2)
    IF(NYP.NE.0)WIGMOD=-WIGMOD
    RETURN
    END    

    SUBROUTINE COE(NMAX,FAC,BI)
    DOUBLE PRECISION FAC,BI
    DIMENSION FAC(0:NMAX),BI(0:NMAX,0:NMAX)
    DO N=0,NMAX
        DO M=0,N/2
            BI(M,N)=FAC(N)/(FAC(M)*FAC(N-M))
            BI(N-M,N)=BI(M,N)
        ENDDO
    ENDDO  
    RETURN
    END
        
    SUBROUTINE COEFREC(MAXCOE,L,NN,NQMAX,PSIP,PSIM)
! CALCULATES THE COEFFICIENTS OF THE RECURSION FORMULA, EQ. (21) IN THE ACCOMPANYING
! CPC PAPER
! MAXCOE EQUALS INT((NQMAX+LMAX)/2)+1 AT CALLS OF THIS ROUTINE.
    DOUBLE PRECISION PSIP,PSIM
    DIMENSION PSIP(MAXCOE,MAXCOE),PSIM(MAXCOE,MAXCOE)
    LP=L+1
    LM=L-1
    DO M1=1,(NQMAX+L)/2+1
        DO M2=1,M1
            M1P2=M1+M2
            NA=M1P2+L
            NNA=NA-2*(NA/2)
            M1M2=M1-M2
! CALCULATION OF PSIP. (IN THIS CASE M1 AND M2 REPRESENT, RESPECTIVELY, L1P AND L2P OR 
! L1P+1 AND L2P+1.)
            IF (NNA.EQ.NN.AND.M1P2.GE.L.AND.M1P2.LE.NQMAX.AND.&
            ABS(M1M2).LE.L) THEN
                IF (M1P2.GT.LP) THEN
                    M1P2L=M1P2+L
                    M1P2ML=M1P2-L
                    PSIP(M1,M2)=SQRT(M1P2L*(M1P2L+1.D0)*M1P2ML*(M1P2ML-1)/((4*M1*M1-1)&
                    *(4*M2*M2-1)))
                ELSE
                    PSIP(M1,M2)=0.D0
                ENDIF
                PSIP(M2,M1)=PSIP(M1,M2) 
            ENDIF
! CALCULATION OF PSIM. (IN THIS CASE M1 AND M2 REPRESENT, RESPECTIVELY, L1P+1 AND L2P
! OR L2P+1 AND L1P.)
            IF (NNA.NE.NN.AND.M1P2.GE.LP.AND.M1P2.LE.NQMAX&
            .AND.M1M2.LE.LP.AND.M1M2.GE.-LM) THEN
                IF (M1M2.LT.L) THEN
                    M1M2L=M1M2+L
                    M1M2ML=M1M2-L 
                    PSIM(M1,M2)=SQRT(M1M2L*(M1M2L+1.D0)*M1M2ML*(M1M2ML-1)/&
                    ((4*M1*M1-1)*(4*M2*M2-1)))
                ELSE
                    PSIM(M1,M2)=0.D0
                ENDIF
                PSIM(M2,M1)=PSIM(M1,M2) 
            ENDIF                                               
        ENDDO
    ENDDO 
    RETURN
    END             
    
