def send_message(M, C):
    from random import randint
    global send_packet

    S = len(M)
    message_bits = M.copy()

    # Identify reliable positions (C[i] == 0)
    reliable_positions = [i for i in range(31) if C[i] == 0]
    unreliable_positions = [i for i in range(31) if C[i] == 1]

    # Number of bits we can send per packet
    bits_per_packet = len(reliable_positions)

    # Number of packets needed
    num_packets = (S + bits_per_packet - 1) // bits_per_packet

    # Send initial packets to help Basma deduce reliable bits
    # Packet 1: All zeros in reliable positions
    packet = [0]*31
    for i in unreliable_positions:
        packet[i] = randint(0, 1)
    send_packet(packet)

    # Packet 2: All ones in reliable positions
    packet = [0]*31
    for i in reliable_positions:
        packet[i] = 1
    for i in unreliable_positions:
        packet[i] = randint(0, 1)
    send_packet(packet)

    # Now send the message chunks
    idx = 0
    while idx < S:
        packet = [0]*31
        # Set message bits in reliable positions
        for i in reliable_positions:
            if idx < S:
                packet[i] = message_bits[idx]
                idx += 1
            else:
                packet[i] = 0  # Padding if message ends
        # Fill unreliable positions with random bits
        for i in unreliable_positions:
            packet[i] = randint(0, 1)
        send_packet(packet)

def receive_message(R):
    S = None  # To be set when we know the message length
    num_packets = len(R)

    # Initialize counts for bits at each position
    bit_counts = [ {0:0, 1:0} for _ in range(31) ]

    # Count occurrences of 0 and 1 at each bit position
    for packet in R:
        for i, bit in enumerate(packet):
            bit_counts[i][bit] += 1

    # Identify reliable bits based on consistency
    reliable_positions = []
    for i in range(31):
        if bit_counts[i][0] == 0 or bit_counts[i][1] == 0:
            reliable_positions.append(i)

    # Deduce message bits from reliable positions
    # Since we don't know where the message bits start, we consider that they start after initial packets
    # We skip the first two packets used to deduce reliable positions

    message_bits = []
    for packet in R[2:]:
        for i in reliable_positions:
            message_bits.append(packet[i])
            if len(message_bits) == S:
                break
        if len(message_bits) == S:
            break
    return message_bits