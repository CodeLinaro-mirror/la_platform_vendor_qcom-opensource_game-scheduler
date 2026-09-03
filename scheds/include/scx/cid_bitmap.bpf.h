/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 *
 * BPF-side helpers for raw, arena-backed cid bitmaps.
 *
 * Unlike struct scx_cmask, these bitmaps carry no range or allocation
 * metadata. Callers must validate @cid and ensure that @bits has storage for
 * it before calling any helper below.
 */
#ifndef __SCX_CID_BITMAP_BPF_H
#define __SCX_CID_BITMAP_BPF_H

#include <lib/arena_map.h>

/*
 * Make the verifier associate every caller with the arena. This is needed when
 * @bits itself was loaded through a global pointer rather than an arena map
 * reference in the calling program.
 */
static __always_inline void __cid_bitmap_touch_arena(void)
{
	asm volatile("" :: "r"(&arena));
}

static __always_inline u64 __arena *__cid_bitmap_word(u64 __arena *bits, u32 cid)
{
	__cid_bitmap_touch_arena();

	return &bits[cid / 64];
}

/**
 * cid_bitmap_test - Test whether a cid is set
 * @bits: arena-backed bitmap to test
 * @cid: cid to test
 *
 * @cid must fit in the caller-provided storage.
 *
 * Return: true if @cid is set, false otherwise.
 */
static __always_inline bool cid_bitmap_test(const u64 __arena *bits, u32 cid)
{
	__cid_bitmap_touch_arena();

	return READ_ONCE(bits[cid / 64]) & (1ULL << (cid & 63));
}

/**
 * cid_bitmap_set - Set a cid without synchronization
 * @bits: arena-backed bitmap to modify
 * @cid: cid to set
 *
 * @cid must fit in the caller-provided storage. The caller must provide
 * synchronization against other writers to the same bitmap word.
 */
static __always_inline void cid_bitmap_set(u64 __arena *bits, u32 cid)
{
	*__cid_bitmap_word(bits, cid) |= 1ULL << (cid & 63);
}

/**
 * cid_bitmap_test_range_all_set - Test whether every cid in a range is set
 * @bits: arena-backed bitmap to test
 * @start: first cid in the range
 * @nr: number of cids in the range
 *
 * The range must fit in the caller-provided storage. An empty range is full.
 * Words are read independently, so the range as a whole is not observed
 * atomically.
 *
 * Return: true if every cid in [@start, @start + @nr) is set, false otherwise.
 */
static __always_inline bool
cid_bitmap_test_range_all_set(const u64 __arena *bits, u32 start, u32 nr)
{
	u32 last, first_wi, last_wi, nr_words, i;

	if (!nr)
		return true;

	__cid_bitmap_touch_arena();

	last = start + nr - 1;
	first_wi = start / 64;
	last_wi = last / 64;
	nr_words = last_wi - first_wi + 1;

	bpf_for(i, 0, nr_words) {
		u32 wi = first_wi + i;
		u64 mask = ~0ULL;

		if (wi == first_wi)
			mask &= ~0ULL << (start & 63);
		if (wi == last_wi)
			mask &= ~0ULL >> (63 - (last & 63));
		if ((READ_ONCE(bits[wi]) & mask) != mask)
			return false;
	}

	return true;
}

/*
 * Atomically assign @set to the bit for @cid.
 *
 * Arena memory takes only the non-fetching read-modify-write forms (see
 * bpf_jit_supports_insn()), which C can't ask for: __sync_fetch_and_or() and
 * friends emit the fetching ones even when the result is dead, and spelling
 * the instruction out in inline asm loses the arena type of the pointer. So
 * the update goes through a compare-and-swap loop, which arena memory does
 * take.
 */
static __always_inline void __cid_bitmap_assign(u64 __arena *bits, u32 cid, bool set)
{
	u64 mask = 1ULL << (cid & 63);
	u64 __arena *word = __cid_bitmap_word(bits, cid);
	u64 old = READ_ONCE(*word);

	while (can_loop) {
		u64 want = set ? (old | mask) : (old & ~mask);
		u64 prev;

		if (want == old)
			return;

		prev = __sync_val_compare_and_swap(word, old, want);
		if (prev == old)
			return;
		old = prev;
	}
}

/**
 * cid_bitmap_set_atomic - Atomically set a cid
 * @bits: arena-backed bitmap to modify
 * @cid: cid to set
 *
 * @cid must fit in the caller-provided storage. The containing word is updated
 * with cmpxchg so concurrent updates to other cids in the word are preserved.
 */
static __always_inline void cid_bitmap_set_atomic(u64 __arena *bits, u32 cid)
{
	__cid_bitmap_assign(bits, cid, true);
}

/**
 * cid_bitmap_clear_atomic - Atomically clear a cid
 * @bits: arena-backed bitmap to modify
 * @cid: cid to clear
 *
 * @cid must fit in the caller-provided storage. The containing word is updated
 * with cmpxchg so concurrent updates to other cids in the word are preserved.
 */
static __always_inline void cid_bitmap_clear_atomic(u64 __arena *bits, u32 cid)
{
	__cid_bitmap_assign(bits, cid, false);
}

/**
 * cid_bitmap_test_and_clear - Atomically test and clear a cid
 * @bits: arena-backed bitmap to modify
 * @cid: cid to test and clear
 *
 * @cid must fit in the caller-provided storage. The containing word is updated
 * with cmpxchg so concurrent updates to other cids in the word are preserved.
 *
 * Return: true if this call cleared @cid, false if it was already clear.
 */
static __always_inline bool cid_bitmap_test_and_clear(u64 __arena *bits, u32 cid)
{
	u64 mask = 1ULL << (cid & 63);
	u64 __arena *word = __cid_bitmap_word(bits, cid);
	u64 old = READ_ONCE(*word);

	while ((old & mask) && can_loop) {
		u64 prev = __sync_val_compare_and_swap(word, old, old & ~mask);

		if (prev == old)
			return true;
		old = prev;
	}

	return false;
}

#endif /* __SCX_CID_BITMAP_BPF_H */
