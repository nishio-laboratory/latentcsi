import numpy as np
import struct

RATE_MCS_MOD_TYPE_POS = 8
RATE_MCS_MOD_TYPE_MSK = 0x7 << RATE_MCS_MOD_TYPE_POS
RATE_MCS_CCK_MSK = 0 << RATE_MCS_MOD_TYPE_POS
RATE_MCS_LEGACY_OFDM_MSK = 1 << RATE_MCS_MOD_TYPE_POS
RATE_MCS_HT_MSK = 2 << RATE_MCS_MOD_TYPE_POS
RATE_MCS_VHT_MSK = 3 << RATE_MCS_MOD_TYPE_POS
RATE_MCS_HE_MSK = 4 << RATE_MCS_MOD_TYPE_POS
RATE_MCS_EHT_MSK = 5 << RATE_MCS_MOD_TYPE_POS
RATE_MCS_CHAN_WIDTH_POS = 11
RATE_MCS_CHAN_WIDTH_MSK = 0x7 << RATE_MCS_CHAN_WIDTH_POS
RATE_MCS_CHAN_WIDTH_20_VAL = 0
RATE_MCS_CHAN_WIDTH_20 = RATE_MCS_CHAN_WIDTH_20_VAL << RATE_MCS_CHAN_WIDTH_POS
RATE_MCS_CHAN_WIDTH_40_VAL = 1
RATE_MCS_CHAN_WIDTH_40 = RATE_MCS_CHAN_WIDTH_40_VAL << RATE_MCS_CHAN_WIDTH_POS
RATE_MCS_CHAN_WIDTH_80_VAL = 2
RATE_MCS_CHAN_WIDTH_80 = RATE_MCS_CHAN_WIDTH_80_VAL << RATE_MCS_CHAN_WIDTH_POS
RATE_MCS_CHAN_WIDTH_160_VAL = 3
RATE_MCS_CHAN_WIDTH_160 = (
    RATE_MCS_CHAN_WIDTH_160_VAL << RATE_MCS_CHAN_WIDTH_POS
)
RATE_MCS_CHAN_WIDTH_320_VAL = 4
RATE_MCS_CHAN_WIDTH_320 = (
    RATE_MCS_CHAN_WIDTH_320_VAL << RATE_MCS_CHAN_WIDTH_POS
)
RATE_HT_MCS_CODE_MSK = 7
RATE_MCS_ANT_A_POS = 14
RATE_MCS_ANT_A_MSK = 1 << RATE_MCS_ANT_A_POS
RATE_MCS_ANT_B_POS = 15
RATE_MCS_ANT_B_MSK = 1 << RATE_MCS_ANT_B_POS
RATE_MCS_LDPC_POS = 16
RATE_MCS_LDPC_MSK = 1 << RATE_MCS_LDPC_POS
RATE_MCS_SS_POS = 16
RATE_MCS_SS_MSK = 1 << RATE_MCS_SS_POS
RATE_MCS_BEAMF_POS = 16
RATE_MCS_BEAMF_MSK = 1 << RATE_MCS_BEAMF_POS


def parseHeader(data):
    header = {}
    header["csi_size"] = struct.unpack("I", data[0:4])[0]
    header["ftm_clock"] = struct.unpack("I", data[8:12])[0]
    header["num_rx"] = data[46]
    header["num_tx"] = data[47]
    header["num_subcarriers"] = struct.unpack("I", data[52:56])[0]
    header["rssi1"] = struct.unpack("I", data[60:64])[0]
    header["rssi2"] = struct.unpack("I", data[64:68])[0]
    header["source_mac"] = struct.unpack("BBBBBB", data[68:74])
    header["source_mac_string"] = (
        "%02x:%02x:%02x:%02x:%02x:%02x" % struct.unpack("BBBBBB", data[68:74])
    )
    header["rate_flags"] = struct.unpack("I", data[92:96])[0]

    rate_format = header["rate_flags"] & RATE_MCS_MOD_TYPE_MSK
    if rate_format == RATE_MCS_CCK_MSK:
        rate_format = "CCK"
    elif rate_format == RATE_MCS_LEGACY_OFDM_MSK:
        rate_format = "LEGACY_OFDM"
    elif rate_format == RATE_MCS_VHT_MSK:
        rate_format = "VHT"
    elif rate_format == RATE_MCS_HT_MSK:
        rate_format = "HT"
    elif rate_format == RATE_MCS_HE_MSK:
        rate_format = "HE"
    elif rate_format == RATE_MCS_EHT_MSK:
        rate_format = "EHT"
    else:
        rate_format = "unknown"
    header["rate_format"] = rate_format

    channel_width = header["rate_flags"] & RATE_MCS_CHAN_WIDTH_MSK
    if channel_width == RATE_MCS_CHAN_WIDTH_20:
        channel_width = "20"
    elif channel_width == RATE_MCS_CHAN_WIDTH_40:
        channel_width = "40"
    elif channel_width == RATE_MCS_CHAN_WIDTH_80:
        channel_width = "80"
    elif channel_width == RATE_MCS_CHAN_WIDTH_160:
        channel_width = "160"
    elif channel_width == RATE_MCS_CHAN_WIDTH_320:
        channel_width = "320"
    else:
        channel_width = "unknown"
    header["channel_width"] = channel_width

    header["mcs"] = header["rate_flags"] & RATE_HT_MCS_CODE_MSK
    header["antenna_a"] = (
        True if header["rate_flags"] & RATE_MCS_ANT_A_MSK else False
    )
    header["antenna_b"] = (
        True if header["rate_flags"] & RATE_MCS_ANT_B_MSK else False
    )
    header["ldpc"] = (
        True if header["rate_flags"] & RATE_MCS_LDPC_MSK else False
    )
    header["ss"] = 2 if header["rate_flags"] & RATE_MCS_SS_MSK else 1
    header["beamforming"] = (
        True if header["rate_flags"] & RATE_MCS_BEAMF_MSK else False
    )

    return header


def parseCsiData(data, header):
    csi_matrix = np.zeros(
        (header["num_subcarriers"], header["num_rx"], header["num_tx"]),
        dtype=complex,
    )
    pos = 0
    for j in range(header["num_rx"]):
        for k in range(header["num_tx"]):
            for n in range(header["num_subcarriers"]):
                real = struct.unpack("h", data[pos : pos + 2])[0]
                imag = struct.unpack("h", data[pos + 2 : pos + 4])[0]
                pos += 4
                csi_matrix[n, j, k] = complex(real, imag)
    return csi_matrix


def parse(input):
    step = 0
    output = []
    while len(input) > step:
        data = {}
        data["header"] = parseHeader(input[step : (step + 272)])
        data["header"]["num_rx"] = 1  # TODO: Try to reproduce RX2
        step += 272
        data["csi_matrix"] = parseCsiData(
            input[step : (step + data["header"]["csi_size"])], data["header"]
        )
        step += data["header"]["csi_size"]
        output.append(data)
    return output
