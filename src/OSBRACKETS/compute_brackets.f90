! Pablo Barham. Jan 2023.
! I hate this stupid language

program compute_brackets
    implicit none
    !dimension fac(0:201),dfac(0:201),defac(0:201),rfac(0:100)
    double precision, allocatable :: BRAC(:,:,:,:,:,:,:,:)
    integer :: np, n1p, mp, n1, n2, n, m, l, istatus, i, l1, l2, eps, l1p, l2p, n2p, nnp, Nq, Nqp, nl1p, nl2p, nn
    double precision :: SI, CO
    
    ! Input parameters (absurdly read from command line, fucking nightmare)
    integer :: NQMAX, LMIN, LMAX
    character(len=25) :: filename, NQMAX_char, LMIN_char, LMAX_char
    CALL get_command_argument(1, filename)
    CALL get_command_argument(2, NQMAX_char)
    CALL get_command_argument(3, LMIN_char)
    CALL get_command_argument(4, LMAX_char)
    read(NQMAX_char, *)NQMAX
    read(LMIN_char, *)LMIN
    read(LMAX_char, *)LMAX

    ! Print the input parameters
    print *, "NQMAX = ", NQMAX
    print *, "LMIN = ", LMIN
    print *, "LMAX = ", LMAX
    
    ! L is what you call lambda
    ! NQMAX is the maximum value of the sum of the "energy" quantum numbers of the two particles: 2 * n1 + 2 * n2 + l1 + l2
    ! Max value of L is l1 + l2. Max value of NQ = 2 * n1 + 2 * n2 + L
    ! So the max value of 2 * (n1 + n2) = NQ - L == NDIF
    !NQMAX = ndif + LMAX

    ! The angular parameters are CO=COS(VARPHI) and SI=SIN(VARPHI). Set, for example (for equal masses)
    SI = 1/sqrt(2.d0)
    CO = SI

    ! Allocate the array
    allocate (BRAC(0:LMAX, 0:(NQMAX-LMIN)/2, 0:(NQMAX-LMIN)/2, 0:(NQMAX-LMIN)/2,&
    0:(NQMAX-LMIN)/2, 0:LMAX, 0:(NQMAX-LMIN)/2, LMIN:LMAX), STAT=istatus)
    BRAC = 0.d0
    
    IF (istatus /= 0) STOP "TESTOSBRAC: allocate(BRAC) failed"

    ! Output is of the form BRAC(N', n1', M', n1, n2, N, M, L)
    CALL ALLOSBRAC(NQMAX, LMIN, LMAX, CO, SI, BRAC)

    ! Write to file
    open(unit=10, file=filename, status='replace', action='write', access='sequential')
    i = 0
    do l = LMIN, LMAX
        do n = 0, LMAX
            do m = 0, (NQMAX-LMIN)/2
                do n1 = 0, (NQMAX-LMIN)/2
                    do n2 = 0, (NQMAX-LMIN)/2
                        do np = 0, LMAX
                            do n1p = 0, (NQMAX-LMIN)/2
                                do mp = 0, (NQMAX-LMIN)/2
                                    i = i + 1

                                    eps = mod((NQMAX - L), 2)
                                    l1 = M + N + eps
                                    l2 = M - N + L
                                    l1p = MP + NP + eps
                                    l2p = MP - NP + L
                                    n2p = M + n1 + n2 - MP - n1p

                                    ! Verify triangle inequality
                                    if ((L > (l1 + l2)) .or. (L < abs(l1 - l2))) then
                                        cycle
                                    end if
                                    
                                    ! Skip unphysical values of n2p, l1, l2, l1p, l2p
                                    if ((n2p < 0) .or. (l1 < 0) .or. (l2 < 0) .or. (l1p < 0) .or. (l2p < 0))then
                                        cycle
                                    end if

                                    ! "Swap" values of N and NP
                                    nnp = (l2p - l1p + L - eps)/2
                                    nn = (l2 - l1 + L - eps)/2
                                    
                                    !if (nnp < 0) then
                                    !     cycle
                                    ! end if

                                    Nq = 2 * n1 + l1 + 2 * n2 + l2
                                    Nqp = 2 * n1p + l1p + 2 * n2p + l2p

                                    !print *, i
                                    ! Write all the indices and the BRAC value to the file
                                    ! Write only if the value is not zero (or close to zero)
                                    ! if (np == 0 .and. n1p == 0 .and. mp == 0 .and. n1 == 0 .and. &
                                    !     n2 == 0 .and. n == 0 .and. m == 1 .and. l == 1) then
                                    !     print *, np, n1p, mp, n1, n2, n, m, l
                                    !     print *, BRAC(np, n1p, mp, n1, n2, n, m, l)
                                    !     print *, BRAC(n, n1, m, n1p, n2p, np, mp, l)
                                    ! end if
                                     
                                    if (l1p == 1 .and. l2p == 1 .and. l1 == 0 .and. l2 == 0 .and. &
                                        n1p == 0 .and. n2p == 1 .and. n1 == 1 .and. n2 == 1 .and. &
                                        L == 0) then
                                        print *, np, n1p, mp, n1, n2, n, m, l
                                        print *, BRAC(np, n1p, mp, n1, n2, n, m, l)
                                        print *, BRAC(n, n1, m, n1p, n2p, np, mp, l)
                                    end if

                                    if (abs(BRAC(np, n1p, mp, n1, n2, n, m, l)) > 1.d-10) then
                                    ! print *, Np, nnp, n2p
                                    ! print *, l1, l2, l1p, l2p
                                    ! print *, M, N, n1, n2, n1p, MP, NP, L
                                    ! print *, Nq, Nqp
                                    ! print *, "-----------------"
                                    !if (abs(BRAC(nnp, n2p, mp, n1, n2, n, m, l)) > 1.d-10) then
                                        
                                        ! To get the original Moshinsky coefficients... ????
                                        write(10, '(8(2I4))', advance="no") np, n1p, mp, n1, n2, n, m, l
                                        write(10, *) BRAC(np, n1p, mp, n1, n2, n, m, l)


                                    end if
                                end do
                            end do
                        end do
                    end do
                end do
            end do
        end do
    end do
    
    deallocate (BRAC,STAT=istatus)

end program compute_brackets