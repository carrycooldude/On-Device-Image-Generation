package com.example.on_device_image_generation.tokenizer

import org.json.JSONObject
import java.io.File
import java.util.Locale

/**
 * CLIP Byte-Pair Encoding (BPE) Tokenizer for Stable Diffusion v1.5, v2.1, and ControlNet Canny.
 *
 * - Supports full `vocab.json` and `merges.txt` when present in the model directory.
 * - Includes a built-in byte-to-unicode table and deterministic token mapping fallback
 *   when `vocab.json` is not separately pushed to the device.
 */
class ClipTokenizer(
    vocabFile: File? = null,
    mergesFile: File? = null
) {
    private val encoder: Map<String, Int>
    private val bpeRanks: Map<Pair<String, String>, Int>
    private val byteEncoder: Map<Int, Char> = buildByteEncoder()
    private val cache = HashMap<String, List<String>>()

    private val whitespaceRegex = Regex("""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", RegexOption.IGNORE_CASE)

    init {
        encoder = if (vocabFile != null && vocabFile.exists()) {
            loadVocabJson(vocabFile)
        } else {
            buildFallbackEncoder()
        }

        bpeRanks = if (mergesFile != null && mergesFile.exists()) {
            loadMerges(mergesFile)
        } else {
            emptyMap()
        }
    }

    /**
     * Encodes text into a fixed-length array of 77 token IDs:
     * `[BOS (49406), ...tokens..., EOS (49407), PAD...]`.
     */
    fun encode(text: String, maxLength: Int = MAX_LENGTH, padWithZero: Boolean = false): IntArray {
        val cleaned = text.lowercase(Locale.US).replace(Regex("""\s+"""), " ").trim()
        val tokens = ArrayList<Int>(maxLength)
        tokens.add(BOS_TOKEN_ID)

        if (cleaned.isNotEmpty()) {
            val matches = whitespaceRegex.findAll(cleaned)
            for (match in matches) {
                val piece = match.value
                val byteEncoded = piece.toByteArray(Charsets.UTF_8)
                    .map { b -> byteEncoder[b.toInt() and 0xFF] ?: ' ' }
                    .joinToString("")

                val bpeTokens = bpe(byteEncoded)
                for (subToken in bpeTokens) {
                    val id = encoder[subToken] ?: fallbackTokenId(subToken)
                    if (tokens.size < maxLength - 1) {
                        tokens.add(id)
                    } else {
                        break
                    }
                }
                if (tokens.size >= maxLength - 1) break
            }
        }

        tokens.add(EOS_TOKEN_ID)
        val padToken = if (padWithZero) 0 else EOS_TOKEN_ID
        while (tokens.size < maxLength) {
            tokens.add(padToken)
        }
        return tokens.toIntArray()
    }

    private fun bpe(token: String): List<String> {
        if (token.isEmpty()) return emptyList()
        cache[token]?.let { return it }

        if (bpeRanks.isEmpty()) {
            val whole = "$token</w>"
            val direct = encoder[whole]
            if (direct != null) {
                val res = listOf(whole)
                cache[token] = res
                return res
            }
            // Split into individual characters + </w> on final char
            val chars = token.mapIndexed { idx, c ->
                if (idx == token.lastIndex) "$c</w>" else c.toString()
            }
            cache[token] = chars
            return chars
        }

        var word = token.mapIndexed { idx, c ->
            if (idx == token.lastIndex) "$c</w>" else c.toString()
        }
        var pairs = getPairs(word)
        if (pairs.isEmpty()) {
            return listOf("$token</w>")
        }

        while (true) {
            val bigram = pairs.minByOrNull { pair -> bpeRanks[pair] ?: Int.MAX_VALUE } ?: break
            if (!bpeRanks.containsKey(bigram)) break

            val (first, second) = bigram
            val newWord = ArrayList<String>()
            var i = 0
            while (i < word.size) {
                val j = word.subList(i, word.size).indexOf(first).let { if (it == -1) -1 else i + it }
                if (j == -1) {
                    newWord.addAll(word.subList(i, word.size))
                    break
                }
                newWord.addAll(word.subList(i, j))
                i = j
                if (word[i] == first && i < word.size - 1 && word[i + 1] == second) {
                    newWord.add(first + second)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i += 1
                }
            }
            word = newWord
            if (word.size == 1) break
            pairs = getPairs(word)
        }

        cache[token] = word
        return word
    }

    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        val pairs = LinkedHashSet<Pair<String, String>>()
        for (i in 0 until word.size - 1) {
            pairs.add(word[i] to word[i + 1])
        }
        return pairs
    }

    private fun loadVocabJson(file: File): Map<String, Int> {
        return try {
            val json = JSONObject(file.readText(Charsets.UTF_8))
            val map = HashMap<String, Int>(json.length())
            val keys = json.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                map[key] = json.getInt(key)
            }
            map
        } catch (_: Exception) {
            buildFallbackEncoder()
        }
    }

    private fun loadMerges(file: File): Map<Pair<String, String>, Int> {
        return try {
            val lines = file.readLines(Charsets.UTF_8)
                .drop(1) // skip version header
                .filter { it.isNotBlank() && !it.startsWith("#") }
            val ranks = HashMap<Pair<String, String>, Int>(lines.size)
            lines.forEachIndexed { index, line ->
                val parts = line.split(" ")
                if (parts.size >= 2) {
                    ranks[parts[0] to parts[1]] = index
                }
            }
            ranks
        } catch (_: Exception) {
            emptyMap()
        }
    }

    private fun fallbackTokenId(subToken: String): Int {
        // Deterministic mapping within valid CLIP vocabulary range [1, 49405]
        var h = 2166136261L
        for (ch in subToken) {
            h = (h xor ch.code.toLong()) * 16777619L
        }
        val positive = (h and 0x7FFFFFFFL).toInt()
        return 256 + (positive % (VOCAB_SIZE - 260))
    }

    private fun buildFallbackEncoder(): Map<String, Int> {
        val map = HashMap<String, Int>(1024)
        byteEncoder.values.forEachIndexed { idx, ch ->
            map[ch.toString()] = idx + 1
            map["$ch</w>"] = idx + 257
        }
        // Common prompt words mapped to standard CLIP token IDs
        val commonClipTokens = mapOf(
            "a</w>" to 320,
            "an</w>" to 550,
            "the</w>" to 518,
            "of</w>" to 539,
            "in</w>" to 530,
            "on</w>" to 525,
            "with</w>" to 593,
            "and</w>" to 537,
            "photo</w>" to 1125,
            "photograph</w>" to 8853,
            "portrait</w>" to 4792,
            "landscape</w>" to 5727,
            "painting</w>" to 3086,
            "digital</w>" to 3618,
            "art</w>" to 794,
            "cinematic</w>" to 25305,
            "lighting</w>" to 5789,
            "highly</w>" to 5363,
            "detailed</w>" to 12159,
            "realistic</w>" to 16053,
            "ultra</w>" to 9706,
            "sharp</w>" to 7816,
            "focus</w>" to 4729,
            "masterpiece</w>" to 22749,
            "futuristic</w>" to 18619,
            "city</w>" to 1615,
            "sunset</w>" to 6024,
            "mountain</w>" to 3972,
            "ocean</w>" to 4802,
            "forest</w>" to 4576,
            "cat</w>" to 2368,
            "dog</w>" to 1929,
            "astronaut</w>" to 18359,
            "cyberpunk</w>" to 26780,
            "studio</w>" to 3499,
            "quality</w>" to 3027,
            "blurry</w>" to 24722,
            "lowres</w>" to 41502,
            "ugly</w>" to 10567,
            "deformed</w>" to 38117,
            ",</w>" to 267,
            ".</w>" to 269
        )
        map.putAll(commonClipTokens)
        map["<|startoftext|>"] = BOS_TOKEN_ID
        map["<|endoftext|>"] = EOS_TOKEN_ID
        return map
    }

    private fun buildByteEncoder(): Map<Int, Char> {
        val bs = ArrayList<Int>()
        for (b in '!'.code..'~'.code) bs.add(b)
        for (b in '¡'.code..'¬'.code) bs.add(b)
        for (b in '®'.code..'ÿ'.code) bs.add(b)
        val cs = ArrayList(bs)
        var n = 0
        for (b in 0..255) {
            if (!bs.contains(b)) {
                bs.add(b)
                cs.add(256 + n)
                n += 1
            }
        }
        val result = HashMap<Int, Char>(256)
        for (i in bs.indices) {
            result[bs[i]] = cs[i].toChar()
        }
        return result
    }

    companion object {
        const val BOS_TOKEN_ID = 49406
        const val EOS_TOKEN_ID = 49407
        const val VOCAB_SIZE = 49408
        const val MAX_LENGTH = 77
    }
}

