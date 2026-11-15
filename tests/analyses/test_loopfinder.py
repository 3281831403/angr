#!/usr/bin/env python3
# pylint: disable=missing-class-docstring,no-self-use
"""
Tests for LoopFinder and LoopSeer on irreducible (multi-entry) loops, as produced by
goto-heavy obfuscated code and firmware, plus a reducible-loop regression test.

All tests run on handwritten amd64 shellcode, so no external binaries are needed.
"""

from __future__ import annotations

__package__ = __package__ or "tests.analyses"

import unittest

import angr
from angr.exploration_techniques import LoopSeer

BASE = 0x1000

# A simple reducible loop with a single entry:
#
#   0x00 A:  xor ecx, ecx
#   0x02 H:  inc al                       <- loop header
#   0x04 C:  add cl, 2 ; cmp cl, 0x40 ; jl H ; jmp X
#   0x15 X:  mov eax, 60 ; xor edi, edi ; syscall   (exit)
#
# After normalization H and C merge into one node with a self-loop back edge.
SHELLCODE_REDUCIBLE = (
    b"\x31\xc9"  # 0x00  xor ecx, ecx
    b"\xfe\xc0"  # 0x02  inc al
    b"\x80\xc1\x02"  # 0x04  add cl, 2
    b"\x80\xf9\x40"  # 0x07  cmp cl, 0x40
    b"\x0f\x8c\xf2\xff\xff\xff"  # 0x0a  jl 0x02
    b"\xe9\x00\x00\x00\x00"  # 0x10  jmp 0x15
    b"\xb8\x3c\x00\x00\x00"  # 0x15  mov eax, 60
    b"\x31\xff"  # 0x1a  xor edi, edi
    b"\x0f\x05"  # 0x1c  syscall
)

# An irreducible loop with two entries, typical of goto-built loops:
#
#   0x00 A1: cmp cl, 0 ; jnz B            cl == 0 -> enter via H, cl != 0 -> enter via M
#   0x09 A2: jmp H
#   0x0e B:  test dl, dl ; jz X ; jmp M
#   0x1b H:  inc al ; test al, al ; js C  <- canonical loop header
#   0x25 M:  dec al ; jmp C               <- second entry
#   0x2c C:  add cl, 2 ; cmp cl, 0x40 ; jl H ; jmp X
#   0x3d X:  mov eax, 60 ; xor edi, edi ; syscall   (exit)
#
# SCC {H, M, C}; entry edges A2->H and B->M; back edge C->H; break edge C->X.
SHELLCODE_TWO_ENTRIES = (
    b"\x80\xf9\x00"  # 0x00  cmp cl, 0
    b"\x0f\x85\x05\x00\x00\x00"  # 0x03  jnz 0x0e
    b"\xe9\x0d\x00\x00\x00"  # 0x09  jmp 0x1b
    b"\x84\xd2"  # 0x0e  test dl, dl
    b"\x0f\x84\x27\x00\x00\x00"  # 0x10  jz 0x3d
    b"\xe9\x0a\x00\x00\x00"  # 0x16  jmp 0x25
    b"\xfe\xc0"  # 0x1b  inc al
    b"\x84\xc0"  # 0x1d  test al, al
    b"\x0f\x88\x07\x00\x00\x00"  # 0x1f  js 0x2c
    b"\xfe\xc8"  # 0x25  dec al
    b"\xe9\x00\x00\x00\x00"  # 0x27  jmp 0x2c
    b"\x80\xc1\x02"  # 0x2c  add cl, 2
    b"\x80\xf9\x40"  # 0x2f  cmp cl, 0x40
    b"\x0f\x8c\xe3\xff\xff\xff"  # 0x32  jl 0x1b
    b"\xe9\x00\x00\x00\x00"  # 0x38  jmp 0x3d
    b"\xb8\x3c\x00\x00\x00"  # 0x3d  mov eax, 60
    b"\x31\xff"  # 0x42  xor edi, edi
    b"\x0f\x05"  # 0x44  syscall
)

# An irreducible loop with two entries and a natural (reducible) loop nested inside:
#
#   0x00 A1: cmp cl, 0 ; jnz B
#   0x09 A2: jmp H
#   0x0e B:  test dl, dl ; jz X ; jmp M
#   0x1b H:  inc al ; test al, al ; js I  <- outer loop header
#   0x25 M:  dec al ; xor ah, ah ; jmp I  <- second entry of the outer loop
#   0x2e I:  inc ah ; cmp ah, 4 ; jge K   <- inner loop header
#   0x39 J:  jmp I                        <- inner back edge
#   0x3e K:  add cl, 2 ; cmp cl, 0x20 ; jl H ; jmp X
#   0x4f X:  mov eax, 60 ; xor edi, edi ; syscall   (exit)
#
# Outer SCC {H, M, I, J, K} with entries H and M; inner SCC {I, J} nested inside.
SHELLCODE_TWO_ENTRIES_NESTED = (
    b"\x80\xf9\x00"  # 0x00  cmp cl, 0
    b"\x0f\x85\x05\x00\x00\x00"  # 0x03  jnz 0x0e
    b"\xe9\x0d\x00\x00\x00"  # 0x09  jmp 0x1b
    b"\x84\xd2"  # 0x0e  test dl, dl
    b"\x0f\x84\x39\x00\x00\x00"  # 0x10  jz 0x4f
    b"\xe9\x0a\x00\x00\x00"  # 0x16  jmp 0x25
    b"\xfe\xc0"  # 0x1b  inc al
    b"\x84\xc0"  # 0x1d  test al, al
    b"\x0f\x88\x09\x00\x00\x00"  # 0x1f  js 0x2e
    b"\xfe\xc8"  # 0x25  dec al
    b"\x30\xe4"  # 0x27  xor ah, ah
    b"\xe9\x00\x00\x00\x00"  # 0x29  jmp 0x2e
    b"\xfe\xc4"  # 0x2e  inc ah
    b"\x80\xfc\x04"  # 0x30  cmp ah, 4
    b"\x0f\x8d\x05\x00\x00\x00"  # 0x33  jge 0x3e
    b"\xe9\xf0\xff\xff\xff"  # 0x39  jmp 0x2e
    b"\x80\xc1\x02"  # 0x3e  add cl, 2
    b"\x80\xf9\x20"  # 0x41  cmp cl, 0x20
    b"\x0f\x8c\xd1\xff\xff\xff"  # 0x44  jl 0x1b
    b"\xe9\x00\x00\x00\x00"  # 0x4a  jmp 0x4f
    b"\xb8\x3c\x00\x00\x00"  # 0x4f  mov eax, 60
    b"\x31\xff"  # 0x54  xor edi, edi
    b"\x0f\x05"  # 0x56  syscall
)


def _project(shellcode: bytes) -> angr.Project:
    return angr.load_shellcode(shellcode, "amd64", start_offset=BASE, load_address=BASE, simos="linux")


def _edge_addrs(edges) -> set[tuple[int, int]]:
    return {(src.addr, dst.addr) for src, dst in edges}


class TestLoopFinder(unittest.TestCase):
    def test_reducible_loop(self):
        # A plain single-entry loop must be reported exactly as before.
        p = _project(SHELLCODE_REDUCIBLE)
        p.analyses.CFGFast(normalize=True)
        lf = p.analyses.LoopFinder(normalize=True)

        assert len(lf.loops) == 1
        loop = lf.loops[0]
        h = BASE + 0x02
        assert loop.entry.addr == h
        assert [n.addr for n in loop.entry_nodes] == [h]
        assert _edge_addrs(loop.entry_edges) == {(BASE + 0x00, h)}
        assert _edge_addrs(loop.continue_edges) == {(h, h)}
        assert _edge_addrs(loop.break_edges) == {(h, BASE + 0x10)}
        assert {n.addr for n in loop.body_nodes} == {h}
        assert loop.subloops == []
        assert lf.loops_hierarchy[BASE] == [loop]

    def test_two_entry_loop(self):
        p = _project(SHELLCODE_TWO_ENTRIES)
        p.analyses.CFGFast(normalize=True)
        lf = p.analyses.LoopFinder(normalize=True)

        assert len(lf.loops) == 1
        loop = lf.loops[0]
        h, m, c = BASE + 0x1B, BASE + 0x25, BASE + 0x2C
        assert loop.entry.addr == h
        assert {n.addr for n in loop.entry_nodes} == {h, m}
        assert _edge_addrs(loop.entry_edges) == {(BASE + 0x09, h), (BASE + 0x16, m)}
        assert _edge_addrs(loop.continue_edges) == {(c, h), (h, m)}
        assert _edge_addrs(loop.break_edges) == {(c, BASE + 0x38)}
        assert {n.addr for n in loop.body_nodes} == {h, m, c}
        assert loop.subloops == []
        assert lf.loops_hierarchy[BASE] == [loop]

    def test_two_entry_loop_with_inner_loop(self):
        p = _project(SHELLCODE_TWO_ENTRIES_NESTED)
        p.analyses.CFGFast(normalize=True)
        lf = p.analyses.LoopFinder(normalize=True)

        assert len(lf.loops) == 2
        outer = next(loop for loop in lf.loops if loop.subloops)
        assert len(outer.subloops) == 1
        inner = outer.subloops[0]
        assert inner in lf.loops

        h, m, i, j, k = (BASE + off for off in (0x1B, 0x25, 0x2E, 0x39, 0x3E))
        # the outer, irreducible loop
        assert outer.entry.addr == h
        assert {n.addr for n in outer.entry_nodes} == {h, m}
        assert _edge_addrs(outer.entry_edges) == {(BASE + 0x09, h), (BASE + 0x16, m)}
        assert _edge_addrs(outer.continue_edges) == {(k, h), (h, m)}
        assert _edge_addrs(outer.break_edges) == {(k, BASE + 0x4A)}
        assert {n.addr for n in outer.body_nodes} == {h, m, i, j, k}
        # the inner, reducible loop nested one level down
        assert inner.entry.addr == i
        assert [n.addr for n in inner.entry_nodes] == [i]
        assert _edge_addrs(inner.entry_edges) == {(h, i), (m, i)}
        assert _edge_addrs(inner.continue_edges) == {(j, i)}
        assert _edge_addrs(inner.break_edges) == {(i, k)}
        assert {n.addr for n in inner.body_nodes} == {i, j}
        assert inner.subloops == []
        # nesting hierarchy: only the outer loop is top-level
        assert lf.loops_hierarchy[BASE] == [outer]


class TestLoopSeerMultiEntry(unittest.TestCase):
    def _run(self, shellcode: bytes, cl: int, outer_header: int, inner_header: int | None = None):
        p = _project(shellcode)
        cfg = p.analyses.CFGFast(normalize=True)

        state = p.factory.entry_state()
        state.regs.rax = 0
        state.regs.rdx = 1
        state.regs.rcx = cl

        simgr = p.factory.simulation_manager(state)
        simgr.use_technique(LoopSeer(cfg=cfg, bound=None))

        while simgr.active:
            simgr.step()
            for s in simgr.active:
                # a loop must never be activated twice on the same path
                active_loops = [loop for loop, _ in s.loop_data.current_loop]
                assert len({id(loop) for loop in active_loops}) == len(active_loops)
                # nested loops must appear outside-in on the current loop stack
                entries = [loop.entry.addr for loop in active_loops]
                assert entries == sorted(entries)
                # trip counts are monotonically non-decreasing along a path
                high_watermarks = s.globals.get("trip_high_watermarks")
                if high_watermarks is None:
                    high_watermarks = {}
                    s.globals["trip_high_watermarks"] = high_watermarks
                for header, counts in s.loop_data.back_edge_trip_counts.items():
                    for idx, count in enumerate(counts):
                        key = (header, idx)
                        assert count >= high_watermarks.get(key, 0)
                        high_watermarks[key] = count

        assert len(simgr.deadended) == 1
        return simgr.deadended[0]

    def test_reducible_loop_trip_count(self):
        d = self._run(SHELLCODE_REDUCIBLE, 0, BASE + 0x02)
        assert d.loop_data.back_edge_trip_counts[BASE + 0x02] == [31]
        assert d.loop_data.header_trip_counts[BASE + 0x02] == [32]
        assert d.loop_data.current_loop == []

    def test_two_entry_loop_trip_count(self):
        # entering through either entry of the same irreducible loop must activate
        # exactly one loop instance and record one trip count per iteration
        for cl in (0, 1):
            d = self._run(SHELLCODE_TWO_ENTRIES, cl, BASE + 0x1B)
            assert d.loop_data.back_edge_trip_counts[BASE + 0x1B] == [31]
            assert d.loop_data.header_trip_counts[BASE + 0x1B] == [32]
            assert d.loop_data.current_loop == []

    def test_two_entry_loop_with_inner_loop_trip_count(self):
        for cl in (0, 1):
            d = self._run(SHELLCODE_TWO_ENTRIES_NESTED, cl, BASE + 0x1B, BASE + 0x2E)
            # outer loop: 16 iterations, one trip count slot
            assert d.loop_data.back_edge_trip_counts[BASE + 0x1B] == [15]
            assert d.loop_data.header_trip_counts[BASE + 0x1B] == [16]
            # inner loop: re-entered on every outer iteration, 4 iterations each time
            assert d.loop_data.back_edge_trip_counts[BASE + 0x2E] == [3] * 16
            assert d.loop_data.header_trip_counts[BASE + 0x2E] == [4] * 16
            assert d.loop_data.current_loop == []


if __name__ == "__main__":
    unittest.main()
