macro_rules! blend_shuffle4 {
    ($a: expr, $b: expr, $imm8: expr) => {
        unsafe {
            match $imm8 & 0xF {
                0x0 => simd_shuffle4($a, $b, [0, 1, 2, 3]),
                0x1 => simd_shuffle4($a, $b, [4, 1, 2, 3]),
                0x2 => simd_shuffle4($a, $b, [0, 5, 2, 3]),
                0x3 => simd_shuffle4($a, $b, [4, 5, 2, 3]),
                0x4 => simd_shuffle4($a, $b, [0, 1, 6, 3]),
                0x5 => simd_shuffle4($a, $b, [4, 1, 6, 3]),
                0x6 => simd_shuffle4($a, $b, [0, 5, 6, 3]),
                0x7 => simd_shuffle4($a, $b, [4, 5, 6, 3]),
                0x8 => simd_shuffle4($a, $b, [0, 1, 2, 7]),
                0x9 => simd_shuffle4($a, $b, [4, 1, 2, 7]),
                0xA => simd_shuffle4($a, $b, [0, 5, 2, 7]),
                0xB => simd_shuffle4($a, $b, [4, 5, 2, 7]),
                0xC => simd_shuffle4($a, $b, [0, 1, 6, 7]),
                0xD => simd_shuffle4($a, $b, [4, 1, 6, 7]),
                0xE => simd_shuffle4($a, $b, [0, 5, 6, 7]),
                0xF => simd_shuffle4($a, $b, [4, 5, 6, 7]),
                _ => unreachable!()
            }
        }
    }
}

macro_rules! fn_imm8_arg2 {
    ($fn_name: expr, $a: expr, $b: expr, $imm8: expr) => {
        unsafe {
            match $imm8 & 0xFF {
                0x00 => $fn_name($a, $b, 0x00),
                0x01 => $fn_name($a, $b, 0x01),
                0x02 => $fn_name($a, $b, 0x02),
                0x03 => $fn_name($a, $b, 0x03),
                0x04 => $fn_name($a, $b, 0x04),
                0x05 => $fn_name($a, $b, 0x05),
                0x06 => $fn_name($a, $b, 0x06),
                0x07 => $fn_name($a, $b, 0x07),
                0x08 => $fn_name($a, $b, 0x08),
                0x09 => $fn_name($a, $b, 0x09),
                0x0A => $fn_name($a, $b, 0x0A),
                0x0B => $fn_name($a, $b, 0x0B),
                0x0C => $fn_name($a, $b, 0x0C),
                0x0D => $fn_name($a, $b, 0x0D),
                0x0E => $fn_name($a, $b, 0x0E),
                0x0F => $fn_name($a, $b, 0x0F),
                0x10 => $fn_name($a, $b, 0x10),
                0x11 => $fn_name($a, $b, 0x11),
                0x12 => $fn_name($a, $b, 0x12),
                0x13 => $fn_name($a, $b, 0x13),
                0x14 => $fn_name($a, $b, 0x14),
                0x15 => $fn_name($a, $b, 0x15),
                0x16 => $fn_name($a, $b, 0x16),
                0x17 => $fn_name($a, $b, 0x17),
                0x18 => $fn_name($a, $b, 0x18),
                0x19 => $fn_name($a, $b, 0x19),
                0x1A => $fn_name($a, $b, 0x1A),
                0x1B => $fn_name($a, $b, 0x1B),
                0x1C => $fn_name($a, $b, 0x1C),
                0x1D => $fn_name($a, $b, 0x1D),
                0x1E => $fn_name($a, $b, 0x1E),
                0x1F => $fn_name($a, $b, 0x1F),
                0x20 => $fn_name($a, $b, 0x20),
                0x21 => $fn_name($a, $b, 0x21),
                0x22 => $fn_name($a, $b, 0x22),
                0x23 => $fn_name($a, $b, 0x23),
                0x24 => $fn_name($a, $b, 0x24),
                0x25 => $fn_name($a, $b, 0x25),
                0x26 => $fn_name($a, $b, 0x26),
                0x27 => $fn_name($a, $b, 0x27),
                0x28 => $fn_name($a, $b, 0x28),
                0x29 => $fn_name($a, $b, 0x29),
                0x2A => $fn_name($a, $b, 0x2A),
                0x2B => $fn_name($a, $b, 0x2B),
                0x2C => $fn_name($a, $b, 0x2C),
                0x2D => $fn_name($a, $b, 0x2D),
                0x2E => $fn_name($a, $b, 0x2E),
                0x2F => $fn_name($a, $b, 0x2F),
                0x30 => $fn_name($a, $b, 0x30),
                0x31 => $fn_name($a, $b, 0x31),
                0x32 => $fn_name($a, $b, 0x32),
                0x33 => $fn_name($a, $b, 0x33),
                0x34 => $fn_name($a, $b, 0x34),
                0x35 => $fn_name($a, $b, 0x35),
                0x36 => $fn_name($a, $b, 0x36),
                0x37 => $fn_name($a, $b, 0x37),
                0x38 => $fn_name($a, $b, 0x38),
                0x39 => $fn_name($a, $b, 0x39),
                0x3A => $fn_name($a, $b, 0x3A),
                0x3B => $fn_name($a, $b, 0x3B),
                0x3C => $fn_name($a, $b, 0x3C),
                0x3D => $fn_name($a, $b, 0x3D),
                0x3E => $fn_name($a, $b, 0x3E),
                0x3F => $fn_name($a, $b, 0x3F),
                0x40 => $fn_name($a, $b, 0x40),
                0x41 => $fn_name($a, $b, 0x41),
                0x42 => $fn_name($a, $b, 0x42),
                0x43 => $fn_name($a, $b, 0x43),
                0x44 => $fn_name($a, $b, 0x44),
                0x45 => $fn_name($a, $b, 0x45),
                0x46 => $fn_name($a, $b, 0x46),
                0x47 => $fn_name($a, $b, 0x47),
                0x48 => $fn_name($a, $b, 0x48),
                0x49 => $fn_name($a, $b, 0x49),
                0x4A => $fn_name($a, $b, 0x4A),
                0x4B => $fn_name($a, $b, 0x4B),
                0x4C => $fn_name($a, $b, 0x4C),
                0x4D => $fn_name($a, $b, 0x4D),
                0x4E => $fn_name($a, $b, 0x4E),
                0x4F => $fn_name($a, $b, 0x4F),
                0x50 => $fn_name($a, $b, 0x50),
                0x51 => $fn_name($a, $b, 0x51),
                0x52 => $fn_name($a, $b, 0x52),
                0x53 => $fn_name($a, $b, 0x53),
                0x54 => $fn_name($a, $b, 0x54),
                0x55 => $fn_name($a, $b, 0x55),
                0x56 => $fn_name($a, $b, 0x56),
                0x57 => $fn_name($a, $b, 0x57),
                0x58 => $fn_name($a, $b, 0x58),
                0x59 => $fn_name($a, $b, 0x59),
                0x5A => $fn_name($a, $b, 0x5A),
                0x5B => $fn_name($a, $b, 0x5B),
                0x5C => $fn_name($a, $b, 0x5C),
                0x5D => $fn_name($a, $b, 0x5D),
                0x5E => $fn_name($a, $b, 0x5E),
                0x5F => $fn_name($a, $b, 0x5F),
                0x60 => $fn_name($a, $b, 0x60),
                0x61 => $fn_name($a, $b, 0x61),
                0x62 => $fn_name($a, $b, 0x62),
                0x63 => $fn_name($a, $b, 0x63),
                0x64 => $fn_name($a, $b, 0x64),
                0x65 => $fn_name($a, $b, 0x65),
                0x66 => $fn_name($a, $b, 0x66),
                0x67 => $fn_name($a, $b, 0x67),
                0x68 => $fn_name($a, $b, 0x68),
                0x69 => $fn_name($a, $b, 0x69),
                0x6A => $fn_name($a, $b, 0x6A),
                0x6B => $fn_name($a, $b, 0x6B),
                0x6C => $fn_name($a, $b, 0x6C),
                0x6D => $fn_name($a, $b, 0x6D),
                0x6E => $fn_name($a, $b, 0x6E),
                0x6F => $fn_name($a, $b, 0x6F),
                0x70 => $fn_name($a, $b, 0x70),
                0x71 => $fn_name($a, $b, 0x71),
                0x72 => $fn_name($a, $b, 0x72),
                0x73 => $fn_name($a, $b, 0x73),
                0x74 => $fn_name($a, $b, 0x74),
                0x75 => $fn_name($a, $b, 0x75),
                0x76 => $fn_name($a, $b, 0x76),
                0x77 => $fn_name($a, $b, 0x77),
                0x78 => $fn_name($a, $b, 0x78),
                0x79 => $fn_name($a, $b, 0x79),
                0x7A => $fn_name($a, $b, 0x7A),
                0x7B => $fn_name($a, $b, 0x7B),
                0x7C => $fn_name($a, $b, 0x7C),
                0x7D => $fn_name($a, $b, 0x7D),
                0x7E => $fn_name($a, $b, 0x7E),
                0x7F => $fn_name($a, $b, 0x7F),
                0x80 => $fn_name($a, $b, 0x80),
                0x81 => $fn_name($a, $b, 0x81),
                0x82 => $fn_name($a, $b, 0x82),
                0x83 => $fn_name($a, $b, 0x83),
                0x84 => $fn_name($a, $b, 0x84),
                0x85 => $fn_name($a, $b, 0x85),
                0x86 => $fn_name($a, $b, 0x86),
                0x87 => $fn_name($a, $b, 0x87),
                0x88 => $fn_name($a, $b, 0x88),
                0x89 => $fn_name($a, $b, 0x89),
                0x8A => $fn_name($a, $b, 0x8A),
                0x8B => $fn_name($a, $b, 0x8B),
                0x8C => $fn_name($a, $b, 0x8C),
                0x8D => $fn_name($a, $b, 0x8D),
                0x8E => $fn_name($a, $b, 0x8E),
                0x8F => $fn_name($a, $b, 0x8F),
                0x90 => $fn_name($a, $b, 0x90),
                0x91 => $fn_name($a, $b, 0x91),
                0x92 => $fn_name($a, $b, 0x92),
                0x93 => $fn_name($a, $b, 0x93),
                0x94 => $fn_name($a, $b, 0x94),
                0x95 => $fn_name($a, $b, 0x95),
                0x96 => $fn_name($a, $b, 0x96),
                0x97 => $fn_name($a, $b, 0x97),
                0x98 => $fn_name($a, $b, 0x98),
                0x99 => $fn_name($a, $b, 0x99),
                0x9A => $fn_name($a, $b, 0x9A),
                0x9B => $fn_name($a, $b, 0x9B),
                0x9C => $fn_name($a, $b, 0x9C),
                0x9D => $fn_name($a, $b, 0x9D),
                0x9E => $fn_name($a, $b, 0x9E),
                0x9F => $fn_name($a, $b, 0x9F),
                0xA0 => $fn_name($a, $b, 0xA0),
                0xA1 => $fn_name($a, $b, 0xA1),
                0xA2 => $fn_name($a, $b, 0xA2),
                0xA3 => $fn_name($a, $b, 0xA3),
                0xA4 => $fn_name($a, $b, 0xA4),
                0xA5 => $fn_name($a, $b, 0xA5),
                0xA6 => $fn_name($a, $b, 0xA6),
                0xA7 => $fn_name($a, $b, 0xA7),
                0xA8 => $fn_name($a, $b, 0xA8),
                0xA9 => $fn_name($a, $b, 0xA9),
                0xAA => $fn_name($a, $b, 0xAA),
                0xAB => $fn_name($a, $b, 0xAB),
                0xAC => $fn_name($a, $b, 0xAC),
                0xAD => $fn_name($a, $b, 0xAD),
                0xAE => $fn_name($a, $b, 0xAE),
                0xAF => $fn_name($a, $b, 0xAF),
                0xB0 => $fn_name($a, $b, 0xB0),
                0xB1 => $fn_name($a, $b, 0xB1),
                0xB2 => $fn_name($a, $b, 0xB2),
                0xB3 => $fn_name($a, $b, 0xB3),
                0xB4 => $fn_name($a, $b, 0xB4),
                0xB5 => $fn_name($a, $b, 0xB5),
                0xB6 => $fn_name($a, $b, 0xB6),
                0xB7 => $fn_name($a, $b, 0xB7),
                0xB8 => $fn_name($a, $b, 0xB8),
                0xB9 => $fn_name($a, $b, 0xB9),
                0xBA => $fn_name($a, $b, 0xBA),
                0xBB => $fn_name($a, $b, 0xBB),
                0xBC => $fn_name($a, $b, 0xBC),
                0xBD => $fn_name($a, $b, 0xBD),
                0xBE => $fn_name($a, $b, 0xBE),
                0xBF => $fn_name($a, $b, 0xBF),
                0xC0 => $fn_name($a, $b, 0xC0),
                0xC1 => $fn_name($a, $b, 0xC1),
                0xC2 => $fn_name($a, $b, 0xC2),
                0xC3 => $fn_name($a, $b, 0xC3),
                0xC4 => $fn_name($a, $b, 0xC4),
                0xC5 => $fn_name($a, $b, 0xC5),
                0xC6 => $fn_name($a, $b, 0xC6),
                0xC7 => $fn_name($a, $b, 0xC7),
                0xC8 => $fn_name($a, $b, 0xC8),
                0xC9 => $fn_name($a, $b, 0xC9),
                0xCA => $fn_name($a, $b, 0xCA),
                0xCB => $fn_name($a, $b, 0xCB),
                0xCC => $fn_name($a, $b, 0xCC),
                0xCD => $fn_name($a, $b, 0xCD),
                0xCE => $fn_name($a, $b, 0xCE),
                0xCF => $fn_name($a, $b, 0xCF),
                0xD0 => $fn_name($a, $b, 0xD0),
                0xD1 => $fn_name($a, $b, 0xD1),
                0xD2 => $fn_name($a, $b, 0xD2),
                0xD3 => $fn_name($a, $b, 0xD3),
                0xD4 => $fn_name($a, $b, 0xD4),
                0xD5 => $fn_name($a, $b, 0xD5),
                0xD6 => $fn_name($a, $b, 0xD6),
                0xD7 => $fn_name($a, $b, 0xD7),
                0xD8 => $fn_name($a, $b, 0xD8),
                0xD9 => $fn_name($a, $b, 0xD9),
                0xDA => $fn_name($a, $b, 0xDA),
                0xDB => $fn_name($a, $b, 0xDB),
                0xDC => $fn_name($a, $b, 0xDC),
                0xDD => $fn_name($a, $b, 0xDD),
                0xDE => $fn_name($a, $b, 0xDE),
                0xDF => $fn_name($a, $b, 0xDF),
                0xE0 => $fn_name($a, $b, 0xE0),
                0xE1 => $fn_name($a, $b, 0xE1),
                0xE2 => $fn_name($a, $b, 0xE2),
                0xE3 => $fn_name($a, $b, 0xE3),
                0xE4 => $fn_name($a, $b, 0xE4),
                0xE5 => $fn_name($a, $b, 0xE5),
                0xE6 => $fn_name($a, $b, 0xE6),
                0xE7 => $fn_name($a, $b, 0xE7),
                0xE8 => $fn_name($a, $b, 0xE8),
                0xE9 => $fn_name($a, $b, 0xE9),
                0xEA => $fn_name($a, $b, 0xEA),
                0xEB => $fn_name($a, $b, 0xEB),
                0xEC => $fn_name($a, $b, 0xEC),
                0xED => $fn_name($a, $b, 0xED),
                0xEE => $fn_name($a, $b, 0xEE),
                0xEF => $fn_name($a, $b, 0xEF),
                0xF0 => $fn_name($a, $b, 0xF0),
                0xF1 => $fn_name($a, $b, 0xF1),
                0xF2 => $fn_name($a, $b, 0xF2),
                0xF3 => $fn_name($a, $b, 0xF3),
                0xF4 => $fn_name($a, $b, 0xF4),
                0xF5 => $fn_name($a, $b, 0xF5),
                0xF6 => $fn_name($a, $b, 0xF6),
                0xF7 => $fn_name($a, $b, 0xF7),
                0xF8 => $fn_name($a, $b, 0xF8),
                0xF9 => $fn_name($a, $b, 0xF9),
                0xFA => $fn_name($a, $b, 0xFA),
                0xFB => $fn_name($a, $b, 0xFB),
                0xFC => $fn_name($a, $b, 0xFC),
                0xFD => $fn_name($a, $b, 0xFD),
                0xFE => $fn_name($a, $b, 0xFE),
                0xFF => $fn_name($a, $b, 0xFF),
                _ => unreachable!()
            }
        }
    }
}

macro_rules! fn_imm8_arg4 {
    ($fn_name: expr, $a: expr, $b: expr, $c: expr, $d: expr, $imm8: expr) => {
        unsafe {
            match $imm8 & 0xFF {
                0x00 => $fn_name($a, $b, $c, $d, 0x00),
                0x01 => $fn_name($a, $b, $c, $d, 0x01),
                0x02 => $fn_name($a, $b, $c, $d, 0x02),
                0x03 => $fn_name($a, $b, $c, $d, 0x03),
                0x04 => $fn_name($a, $b, $c, $d, 0x04),
                0x05 => $fn_name($a, $b, $c, $d, 0x05),
                0x06 => $fn_name($a, $b, $c, $d, 0x06),
                0x07 => $fn_name($a, $b, $c, $d, 0x07),
                0x08 => $fn_name($a, $b, $c, $d, 0x08),
                0x09 => $fn_name($a, $b, $c, $d, 0x09),
                0x0A => $fn_name($a, $b, $c, $d, 0x0A),
                0x0B => $fn_name($a, $b, $c, $d, 0x0B),
                0x0C => $fn_name($a, $b, $c, $d, 0x0C),
                0x0D => $fn_name($a, $b, $c, $d, 0x0D),
                0x0E => $fn_name($a, $b, $c, $d, 0x0E),
                0x0F => $fn_name($a, $b, $c, $d, 0x0F),
                0x10 => $fn_name($a, $b, $c, $d, 0x10),
                0x11 => $fn_name($a, $b, $c, $d, 0x11),
                0x12 => $fn_name($a, $b, $c, $d, 0x12),
                0x13 => $fn_name($a, $b, $c, $d, 0x13),
                0x14 => $fn_name($a, $b, $c, $d, 0x14),
                0x15 => $fn_name($a, $b, $c, $d, 0x15),
                0x16 => $fn_name($a, $b, $c, $d, 0x16),
                0x17 => $fn_name($a, $b, $c, $d, 0x17),
                0x18 => $fn_name($a, $b, $c, $d, 0x18),
                0x19 => $fn_name($a, $b, $c, $d, 0x19),
                0x1A => $fn_name($a, $b, $c, $d, 0x1A),
                0x1B => $fn_name($a, $b, $c, $d, 0x1B),
                0x1C => $fn_name($a, $b, $c, $d, 0x1C),
                0x1D => $fn_name($a, $b, $c, $d, 0x1D),
                0x1E => $fn_name($a, $b, $c, $d, 0x1E),
                0x1F => $fn_name($a, $b, $c, $d, 0x1F),
                0x20 => $fn_name($a, $b, $c, $d, 0x20),
                0x21 => $fn_name($a, $b, $c, $d, 0x21),
                0x22 => $fn_name($a, $b, $c, $d, 0x22),
                0x23 => $fn_name($a, $b, $c, $d, 0x23),
                0x24 => $fn_name($a, $b, $c, $d, 0x24),
                0x25 => $fn_name($a, $b, $c, $d, 0x25),
                0x26 => $fn_name($a, $b, $c, $d, 0x26),
                0x27 => $fn_name($a, $b, $c, $d, 0x27),
                0x28 => $fn_name($a, $b, $c, $d, 0x28),
                0x29 => $fn_name($a, $b, $c, $d, 0x29),
                0x2A => $fn_name($a, $b, $c, $d, 0x2A),
                0x2B => $fn_name($a, $b, $c, $d, 0x2B),
                0x2C => $fn_name($a, $b, $c, $d, 0x2C),
                0x2D => $fn_name($a, $b, $c, $d, 0x2D),
                0x2E => $fn_name($a, $b, $c, $d, 0x2E),
                0x2F => $fn_name($a, $b, $c, $d, 0x2F),
                0x30 => $fn_name($a, $b, $c, $d, 0x30),
                0x31 => $fn_name($a, $b, $c, $d, 0x31),
                0x32 => $fn_name($a, $b, $c, $d, 0x32),
                0x33 => $fn_name($a, $b, $c, $d, 0x33),
                0x34 => $fn_name($a, $b, $c, $d, 0x34),
                0x35 => $fn_name($a, $b, $c, $d, 0x35),
                0x36 => $fn_name($a, $b, $c, $d, 0x36),
                0x37 => $fn_name($a, $b, $c, $d, 0x37),
                0x38 => $fn_name($a, $b, $c, $d, 0x38),
                0x39 => $fn_name($a, $b, $c, $d, 0x39),
                0x3A => $fn_name($a, $b, $c, $d, 0x3A),
                0x3B => $fn_name($a, $b, $c, $d, 0x3B),
                0x3C => $fn_name($a, $b, $c, $d, 0x3C),
                0x3D => $fn_name($a, $b, $c, $d, 0x3D),
                0x3E => $fn_name($a, $b, $c, $d, 0x3E),
                0x3F => $fn_name($a, $b, $c, $d, 0x3F),
                0x40 => $fn_name($a, $b, $c, $d, 0x40),
                0x41 => $fn_name($a, $b, $c, $d, 0x41),
                0x42 => $fn_name($a, $b, $c, $d, 0x42),
                0x43 => $fn_name($a, $b, $c, $d, 0x43),
                0x44 => $fn_name($a, $b, $c, $d, 0x44),
                0x45 => $fn_name($a, $b, $c, $d, 0x45),
                0x46 => $fn_name($a, $b, $c, $d, 0x46),
                0x47 => $fn_name($a, $b, $c, $d, 0x47),
                0x48 => $fn_name($a, $b, $c, $d, 0x48),
                0x49 => $fn_name($a, $b, $c, $d, 0x49),
                0x4A => $fn_name($a, $b, $c, $d, 0x4A),
                0x4B => $fn_name($a, $b, $c, $d, 0x4B),
                0x4C => $fn_name($a, $b, $c, $d, 0x4C),
                0x4D => $fn_name($a, $b, $c, $d, 0x4D),
                0x4E => $fn_name($a, $b, $c, $d, 0x4E),
                0x4F => $fn_name($a, $b, $c, $d, 0x4F),
                0x50 => $fn_name($a, $b, $c, $d, 0x50),
                0x51 => $fn_name($a, $b, $c, $d, 0x51),
                0x52 => $fn_name($a, $b, $c, $d, 0x52),
                0x53 => $fn_name($a, $b, $c, $d, 0x53),
                0x54 => $fn_name($a, $b, $c, $d, 0x54),
                0x55 => $fn_name($a, $b, $c, $d, 0x55),
                0x56 => $fn_name($a, $b, $c, $d, 0x56),
                0x57 => $fn_name($a, $b, $c, $d, 0x57),
                0x58 => $fn_name($a, $b, $c, $d, 0x58),
                0x59 => $fn_name($a, $b, $c, $d, 0x59),
                0x5A => $fn_name($a, $b, $c, $d, 0x5A),
                0x5B => $fn_name($a, $b, $c, $d, 0x5B),
                0x5C => $fn_name($a, $b, $c, $d, 0x5C),
                0x5D => $fn_name($a, $b, $c, $d, 0x5D),
                0x5E => $fn_name($a, $b, $c, $d, 0x5E),
                0x5F => $fn_name($a, $b, $c, $d, 0x5F),
                0x60 => $fn_name($a, $b, $c, $d, 0x60),
                0x61 => $fn_name($a, $b, $c, $d, 0x61),
                0x62 => $fn_name($a, $b, $c, $d, 0x62),
                0x63 => $fn_name($a, $b, $c, $d, 0x63),
                0x64 => $fn_name($a, $b, $c, $d, 0x64),
                0x65 => $fn_name($a, $b, $c, $d, 0x65),
                0x66 => $fn_name($a, $b, $c, $d, 0x66),
                0x67 => $fn_name($a, $b, $c, $d, 0x67),
                0x68 => $fn_name($a, $b, $c, $d, 0x68),
                0x69 => $fn_name($a, $b, $c, $d, 0x69),
                0x6A => $fn_name($a, $b, $c, $d, 0x6A),
                0x6B => $fn_name($a, $b, $c, $d, 0x6B),
                0x6C => $fn_name($a, $b, $c, $d, 0x6C),
                0x6D => $fn_name($a, $b, $c, $d, 0x6D),
                0x6E => $fn_name($a, $b, $c, $d, 0x6E),
                0x6F => $fn_name($a, $b, $c, $d, 0x6F),
                0x70 => $fn_name($a, $b, $c, $d, 0x70),
                0x71 => $fn_name($a, $b, $c, $d, 0x71),
                0x72 => $fn_name($a, $b, $c, $d, 0x72),
                0x73 => $fn_name($a, $b, $c, $d, 0x73),
                0x74 => $fn_name($a, $b, $c, $d, 0x74),
                0x75 => $fn_name($a, $b, $c, $d, 0x75),
                0x76 => $fn_name($a, $b, $c, $d, 0x76),
                0x77 => $fn_name($a, $b, $c, $d, 0x77),
                0x78 => $fn_name($a, $b, $c, $d, 0x78),
                0x79 => $fn_name($a, $b, $c, $d, 0x79),
                0x7A => $fn_name($a, $b, $c, $d, 0x7A),
                0x7B => $fn_name($a, $b, $c, $d, 0x7B),
                0x7C => $fn_name($a, $b, $c, $d, 0x7C),
                0x7D => $fn_name($a, $b, $c, $d, 0x7D),
                0x7E => $fn_name($a, $b, $c, $d, 0x7E),
                0x7F => $fn_name($a, $b, $c, $d, 0x7F),
                0x80 => $fn_name($a, $b, $c, $d, 0x80),
                0x81 => $fn_name($a, $b, $c, $d, 0x81),
                0x82 => $fn_name($a, $b, $c, $d, 0x82),
                0x83 => $fn_name($a, $b, $c, $d, 0x83),
                0x84 => $fn_name($a, $b, $c, $d, 0x84),
                0x85 => $fn_name($a, $b, $c, $d, 0x85),
                0x86 => $fn_name($a, $b, $c, $d, 0x86),
                0x87 => $fn_name($a, $b, $c, $d, 0x87),
                0x88 => $fn_name($a, $b, $c, $d, 0x88),
                0x89 => $fn_name($a, $b, $c, $d, 0x89),
                0x8A => $fn_name($a, $b, $c, $d, 0x8A),
                0x8B => $fn_name($a, $b, $c, $d, 0x8B),
                0x8C => $fn_name($a, $b, $c, $d, 0x8C),
                0x8D => $fn_name($a, $b, $c, $d, 0x8D),
                0x8E => $fn_name($a, $b, $c, $d, 0x8E),
                0x8F => $fn_name($a, $b, $c, $d, 0x8F),
                0x90 => $fn_name($a, $b, $c, $d, 0x90),
                0x91 => $fn_name($a, $b, $c, $d, 0x91),
                0x92 => $fn_name($a, $b, $c, $d, 0x92),
                0x93 => $fn_name($a, $b, $c, $d, 0x93),
                0x94 => $fn_name($a, $b, $c, $d, 0x94),
                0x95 => $fn_name($a, $b, $c, $d, 0x95),
                0x96 => $fn_name($a, $b, $c, $d, 0x96),
                0x97 => $fn_name($a, $b, $c, $d, 0x97),
                0x98 => $fn_name($a, $b, $c, $d, 0x98),
                0x99 => $fn_name($a, $b, $c, $d, 0x99),
                0x9A => $fn_name($a, $b, $c, $d, 0x9A),
                0x9B => $fn_name($a, $b, $c, $d, 0x9B),
                0x9C => $fn_name($a, $b, $c, $d, 0x9C),
                0x9D => $fn_name($a, $b, $c, $d, 0x9D),
                0x9E => $fn_name($a, $b, $c, $d, 0x9E),
                0x9F => $fn_name($a, $b, $c, $d, 0x9F),
                0xA0 => $fn_name($a, $b, $c, $d, 0xA0),
                0xA1 => $fn_name($a, $b, $c, $d, 0xA1),
                0xA2 => $fn_name($a, $b, $c, $d, 0xA2),
                0xA3 => $fn_name($a, $b, $c, $d, 0xA3),
                0xA4 => $fn_name($a, $b, $c, $d, 0xA4),
                0xA5 => $fn_name($a, $b, $c, $d, 0xA5),
                0xA6 => $fn_name($a, $b, $c, $d, 0xA6),
                0xA7 => $fn_name($a, $b, $c, $d, 0xA7),
                0xA8 => $fn_name($a, $b, $c, $d, 0xA8),
                0xA9 => $fn_name($a, $b, $c, $d, 0xA9),
                0xAA => $fn_name($a, $b, $c, $d, 0xAA),
                0xAB => $fn_name($a, $b, $c, $d, 0xAB),
                0xAC => $fn_name($a, $b, $c, $d, 0xAC),
                0xAD => $fn_name($a, $b, $c, $d, 0xAD),
                0xAE => $fn_name($a, $b, $c, $d, 0xAE),
                0xAF => $fn_name($a, $b, $c, $d, 0xAF),
                0xB0 => $fn_name($a, $b, $c, $d, 0xB0),
                0xB1 => $fn_name($a, $b, $c, $d, 0xB1),
                0xB2 => $fn_name($a, $b, $c, $d, 0xB2),
                0xB3 => $fn_name($a, $b, $c, $d, 0xB3),
                0xB4 => $fn_name($a, $b, $c, $d, 0xB4),
                0xB5 => $fn_name($a, $b, $c, $d, 0xB5),
                0xB6 => $fn_name($a, $b, $c, $d, 0xB6),
                0xB7 => $fn_name($a, $b, $c, $d, 0xB7),
                0xB8 => $fn_name($a, $b, $c, $d, 0xB8),
                0xB9 => $fn_name($a, $b, $c, $d, 0xB9),
                0xBA => $fn_name($a, $b, $c, $d, 0xBA),
                0xBB => $fn_name($a, $b, $c, $d, 0xBB),
                0xBC => $fn_name($a, $b, $c, $d, 0xBC),
                0xBD => $fn_name($a, $b, $c, $d, 0xBD),
                0xBE => $fn_name($a, $b, $c, $d, 0xBE),
                0xBF => $fn_name($a, $b, $c, $d, 0xBF),
                0xC0 => $fn_name($a, $b, $c, $d, 0xC0),
                0xC1 => $fn_name($a, $b, $c, $d, 0xC1),
                0xC2 => $fn_name($a, $b, $c, $d, 0xC2),
                0xC3 => $fn_name($a, $b, $c, $d, 0xC3),
                0xC4 => $fn_name($a, $b, $c, $d, 0xC4),
                0xC5 => $fn_name($a, $b, $c, $d, 0xC5),
                0xC6 => $fn_name($a, $b, $c, $d, 0xC6),
                0xC7 => $fn_name($a, $b, $c, $d, 0xC7),
                0xC8 => $fn_name($a, $b, $c, $d, 0xC8),
                0xC9 => $fn_name($a, $b, $c, $d, 0xC9),
                0xCA => $fn_name($a, $b, $c, $d, 0xCA),
                0xCB => $fn_name($a, $b, $c, $d, 0xCB),
                0xCC => $fn_name($a, $b, $c, $d, 0xCC),
                0xCD => $fn_name($a, $b, $c, $d, 0xCD),
                0xCE => $fn_name($a, $b, $c, $d, 0xCE),
                0xCF => $fn_name($a, $b, $c, $d, 0xCF),
                0xD0 => $fn_name($a, $b, $c, $d, 0xD0),
                0xD1 => $fn_name($a, $b, $c, $d, 0xD1),
                0xD2 => $fn_name($a, $b, $c, $d, 0xD2),
                0xD3 => $fn_name($a, $b, $c, $d, 0xD3),
                0xD4 => $fn_name($a, $b, $c, $d, 0xD4),
                0xD5 => $fn_name($a, $b, $c, $d, 0xD5),
                0xD6 => $fn_name($a, $b, $c, $d, 0xD6),
                0xD7 => $fn_name($a, $b, $c, $d, 0xD7),
                0xD8 => $fn_name($a, $b, $c, $d, 0xD8),
                0xD9 => $fn_name($a, $b, $c, $d, 0xD9),
                0xDA => $fn_name($a, $b, $c, $d, 0xDA),
                0xDB => $fn_name($a, $b, $c, $d, 0xDB),
                0xDC => $fn_name($a, $b, $c, $d, 0xDC),
                0xDD => $fn_name($a, $b, $c, $d, 0xDD),
                0xDE => $fn_name($a, $b, $c, $d, 0xDE),
                0xDF => $fn_name($a, $b, $c, $d, 0xDF),
                0xE0 => $fn_name($a, $b, $c, $d, 0xE0),
                0xE1 => $fn_name($a, $b, $c, $d, 0xE1),
                0xE2 => $fn_name($a, $b, $c, $d, 0xE2),
                0xE3 => $fn_name($a, $b, $c, $d, 0xE3),
                0xE4 => $fn_name($a, $b, $c, $d, 0xE4),
                0xE5 => $fn_name($a, $b, $c, $d, 0xE5),
                0xE6 => $fn_name($a, $b, $c, $d, 0xE6),
                0xE7 => $fn_name($a, $b, $c, $d, 0xE7),
                0xE8 => $fn_name($a, $b, $c, $d, 0xE8),
                0xE9 => $fn_name($a, $b, $c, $d, 0xE9),
                0xEA => $fn_name($a, $b, $c, $d, 0xEA),
                0xEB => $fn_name($a, $b, $c, $d, 0xEB),
                0xEC => $fn_name($a, $b, $c, $d, 0xEC),
                0xED => $fn_name($a, $b, $c, $d, 0xED),
                0xEE => $fn_name($a, $b, $c, $d, 0xEE),
                0xEF => $fn_name($a, $b, $c, $d, 0xEF),
                0xF0 => $fn_name($a, $b, $c, $d, 0xF0),
                0xF1 => $fn_name($a, $b, $c, $d, 0xF1),
                0xF2 => $fn_name($a, $b, $c, $d, 0xF2),
                0xF3 => $fn_name($a, $b, $c, $d, 0xF3),
                0xF4 => $fn_name($a, $b, $c, $d, 0xF4),
                0xF5 => $fn_name($a, $b, $c, $d, 0xF5),
                0xF6 => $fn_name($a, $b, $c, $d, 0xF6),
                0xF7 => $fn_name($a, $b, $c, $d, 0xF7),
                0xF8 => $fn_name($a, $b, $c, $d, 0xF8),
                0xF9 => $fn_name($a, $b, $c, $d, 0xF9),
                0xFA => $fn_name($a, $b, $c, $d, 0xFA),
                0xFB => $fn_name($a, $b, $c, $d, 0xFB),
                0xFC => $fn_name($a, $b, $c, $d, 0xFC),
                0xFD => $fn_name($a, $b, $c, $d, 0xFD),
                0xFE => $fn_name($a, $b, $c, $d, 0xFE),
                0xFF => $fn_name($a, $b, $c, $d, 0xFF),
                _ => unreachable!()
            }
        }
    }
}
