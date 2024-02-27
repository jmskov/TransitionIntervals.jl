# distance functions

function additive_noise_distances(image, target)
    buffer = zeros(4)
    return additive_noise_distances!(image, target, buffer)
end

function additive_noise_distances!(image, target, buffer)
    return additive_noise_distances!(image[1], image[2], target[1], target[2], buffer)
end

function additive_noise_distances!(C, D, A, B, buffer)
    W = D - C

    Δ1 = abs(B-C)
    Δ2 = -abs(A-D)
    # check if target contains image
    if A <= C <= D <= B
        Δ3 = abs(B-D)
        Δ4 = -abs(A-C)
    # check if image is larger than target
    elseif C <= A <= B <= D
        Δ3 = 0  
        Δ4 = 0
    else
        if D > B # image UB is greather than state UB 
            if !(A <= C <= B)
                Δ1 *= -1
            end
            Δ3 = -abs(B-D)
            Δ4 = -abs(A+W-D)
        else
            if !(A <= D <= B)
                Δ2 *= -1
            end
            Δ3 = abs(B-W-C)
            Δ4 = abs(A-C) 
        end
        if Δ4 > Δ3 
            Δ3 = 0
            Δ4 = 0
        end
    end
    
    buffer[1] = Δ1
    buffer[2] = Δ2
    buffer[3] = Δ3
    buffer[4] = Δ4
    return buffer
end

function multiplicative_noise_distances(image, target)
    A, B = target
    C, D = image

    # let's see how this ai generated code works...
    # Todo: Δ1 can also be things not Inf possibly...
    if C > 0 && D > 0
        Δ1 = B / C
    elseif C < 0 && D < 0
        Δ1 = A / D
    else
        # if B/C <= 0 && A / D <= 0
        Δ1 = Inf
        # else
            # Δ1 = max(B/C, A/D)
        # end
    end
    Δ2 = ifelse(C > 0 && D > 0, A/D, ifelse(C < 0 && D < 0, B/C, 0))
    # Δ4 = ifelse(C > 0 && D > 0, A/C, ifelse(D < 0 && C < 0, B/D, 0))
    # Δ3 = ifelse(C > 0 && D > 0, B/D, ifelse(C < 0 && D < 0, A/C, min(max(A/C, 0), max(B/D, 0))))

    Δ3 = B/D
    Δ4 = A/C
    
    if Δ4 > Δ3
        Δ3 = 0
        Δ4 = 0
    end

    if Δ2 > Δ1
        Δ1 = Inf
        Δ2 = 0
    end

    # check for NaN...
    if isnan(Δ1)
        Δ1 = Inf
    end
    if isnan(Δ2)
        Δ2 = 0
    end
    if isnan(Δ3)
        Δ3 = 0
    end
    if isnan(Δ4)
        Δ4 = 0
    end
    
    Δ1 = Inf
    # Δ2 = 0
    # Δ3 = 0
    # Δ4 = 0

    @assert Δ2 <= Δ1
    @assert Δ4 <= Δ3
    return Δ1, Δ2, Δ3, Δ4
end
