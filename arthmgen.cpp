// addgen.cpp — Génère un corpus d’arithmétique propre ("A + B = C\n" / "A - B = C\n").
// - Format STRICT: espaces autour de l’opérateur et de '=' ; une ligne par exemple.
// - Opérations: additions, soustractions, ou les deux (--ops add|sub|both).
// - Modes: grid (toutes paires) | random (échantillonné).
// - Options: plage, nombre, ordre, unicité (par op), non-négativité pour sub, balance des ops, seed, shuffle, chemin.
//
// Compile: g++ -O3 -std=c++17 addgen.cpp -o addgen
// Windows: cl /O2 /std:c++17 addgen.cpp

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

enum class Op { Add, Sub };

struct Sample { int a, b; Op op; };

struct Args {
    int minVal = 1;
    int maxVal = 99;
    std::string mode = "random";   // "grid" | "random"
    int count = 10000;             // utilisé si mode=random
    bool bothOrders = true;        // additions: inclure (a,b) ET (b,a). sub: ignoré si --nonneg=1
    bool uniquePairs = false;      // unicité par triplet (op,a,b) en random
    bool shuffleOut = true;        // mélanger la sortie
    uint64_t seed = 0;             // 0 => auto seed
    std::string ops = "both";      // "add" | "sub" | "both"
    bool nonneg = true;            // sub: interdire résultats négatifs (a-b >= 0)
    bool balance = true;           // random + ops=both: équilibrer add/sub
    std::string outPath = "arith.txt";
};

static void usage(const char* prog) {
    std::cerr <<
        "Usage:\n"
        "  " << prog << " [--min 1] [--max 99] [--mode grid|random]\n"
        "               [--ops add|sub|both] [--nonneg 0|1]\n"
        "               [--count 10000] [--both-orders 0|1] [--unique 0|1]\n"
        "               [--balance 0|1] [--shuffle 0|1]\n"
        "               [--seed N] [--out arith.txt]\n\n"
        "Notes:\n"
        "  * Format strict: \"A + B = C\\n\" ou \"A - B = C\\n\"\n"
        "  * --nonneg=1 (defaut): pour la soustraction, on n’écrit que a-b avec a>=b.\n"
        "  * --both-orders s’applique aux additions; pour sub avec --nonneg=1, l’ordre inverse est négatif, donc ignoré.\n"
        "  * --unique garantit l’unicité par (op,a,b) en mode random.\n\n"
        "Exemples:\n"
        "  # Grille 1..9, additions et soustractions (non négatives), mélangée\n"
        "  " << prog << " --min 1 --max 9 --mode grid --ops both --nonneg 1 --shuffle 1 --out arith_1_9.txt\n\n"
        "  # 50k exemples aléatoires équilibrés add/sub sur 0..99 (non négatif pour sub)\n"
        "  " << prog << " --min 0 --max 99 --mode random --ops both --count 50000 --balance 1 --nonneg 1 --unique 1 --out arith_rand.txt\n";
}

static bool parseInt(const std::string& s, int& out) {
    try { size_t p = 0; long v = std::stol(s, &p, 10); if (p != s.size()) return false; out = (int)v; return true; }
    catch (...) { return false; }
}
static bool parseU64(const std::string& s, uint64_t& out) {
    try { size_t p = 0; unsigned long long v = std::stoull(s, &p, 10); if (p != s.size()) return false; out = (uint64_t)v; return true; }
    catch (...) { return false; }
}

static Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto need = [&](const char* name)->std::string {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << name << "\n"; usage(argv[0]); std::exit(1); }
            return std::string(argv[++i]);
            };
        if (k == "--min") { if (!parseInt(need("--min"), a.minVal)) { std::cerr << "Bad --min\n"; std::exit(1); } }
        else if (k == "--max") { if (!parseInt(need("--max"), a.maxVal)) { std::cerr << "Bad --max\n"; std::exit(1); } }
        else if (k == "--mode") { a.mode = need("--mode"); }
        else if (k == "--count") { if (!parseInt(need("--count"), a.count)) { std::cerr << "Bad --count\n"; std::exit(1); } }
        else if (k == "--both-orders") { int v; if (!parseInt(need("--both-orders"), v)) { std::cerr << "Bad --both-orders\n"; std::exit(1); } a.bothOrders = (v != 0); }
        else if (k == "--unique") { int v; if (!parseInt(need("--unique"), v)) { std::cerr << "Bad --unique\n"; std::exit(1); } a.uniquePairs = (v != 0); }
        else if (k == "--shuffle") { int v; if (!parseInt(need("--shuffle"), v)) { std::cerr << "Bad --shuffle\n"; std::exit(1); } a.shuffleOut = (v != 0); }
        else if (k == "--seed") { if (!parseU64(need("--seed"), a.seed)) { std::cerr << "Bad --seed\n"; std::exit(1); } }
        else if (k == "--out") { a.outPath = need("--out"); }
        else if (k == "--ops") { a.ops = need("--ops"); }
        else if (k == "--nonneg") { int v; if (!parseInt(need("--nonneg"), v)) { std::cerr << "Bad --nonneg\n"; std::exit(1); } a.nonneg = (v != 0); }
        else if (k == "--balance") { int v; if (!parseInt(need("--balance"), v)) { std::cerr << "Bad --balance\n"; std::exit(1); } a.balance = (v != 0); }
        else if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else { std::cerr << "Unknown arg: " << k << "\n"; usage(argv[0]); std::exit(1); }
    }
    if (a.minVal > a.maxVal) std::swap(a.minVal, a.maxVal);
    if (a.mode != "grid" && a.mode != "random") {
        std::cerr << "--mode must be 'grid' or 'random'\n"; std::exit(1);
    }
    if (a.mode == "random" && a.count <= 0) {
        std::cerr << "--count must be > 0 for random mode\n"; std::exit(1);
    }
    // sanitize ops
    if (a.ops != "add" && a.ops != "sub" && a.ops != "both") {
        std::cerr << "--ops must be 'add', 'sub' or 'both'\n"; std::exit(1);
    }
    return a;
}

static inline std::string key_of(const Sample& s) {
    char op = (s.op == Op::Add) ? '+' : '-';
    return std::string(1, op) + ":" + std::to_string(s.a) + "," + std::to_string(s.b);
}

int main(int argc, char** argv) {
    auto opt = parseArgs(argc, argv);
    const bool wantAdd = (opt.ops == "add" || opt.ops == "both");
    const bool wantSub = (opt.ops == "sub" || opt.ops == "both");

    // RNG
    uint64_t seed = opt.seed ? opt.seed
        : (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> uni(opt.minVal, opt.maxVal);

    std::vector<Sample> samples;
    samples.reserve((size_t)opt.count);

    auto push_add = [&](int a, int b) {
        if (opt.bothOrders) {
            samples.push_back({ a, b, Op::Add });
            if (a != b) samples.push_back({ b, a, Op::Add });
        }
        else {
            if (a <= b) samples.push_back({ a, b, Op::Add });
        }
        };

    auto push_sub = [&](int a, int b) {
        if (opt.nonneg) {
            if (a >= b) samples.push_back({ a, b, Op::Sub });
            // si bothOrders=true, l’ordre inverse serait négatif: on l’ignore volontairement
        }
        else {
            if (opt.bothOrders) {
                samples.push_back({ a, b, Op::Sub });
                if (a != b) samples.push_back({ b, a, Op::Sub });
            }
            else {
                if (a >= b) samples.push_back({ a, b, Op::Sub }); // éviter doublons
            }
        }
        };

    if (opt.mode == "grid") {
        for (int a = opt.minVal; a <= opt.maxVal; ++a) {
            for (int b = opt.minVal; b <= opt.maxVal; ++b) {
                if (wantAdd) push_add(a, b);
                if (wantSub) push_sub(a, b);
            }
        }
        if (opt.shuffleOut) std::shuffle(samples.begin(), samples.end(), rng);
    }
    else {
        // random
        const int domainSize = (opt.maxVal - opt.minVal + 1) * (opt.maxVal - opt.minVal + 1);
        // Unicité gérée via (op,a,b)
        std::unordered_set<std::string> seen;
        seen.reserve((size_t)opt.count * 2);

        auto try_insert = [&](const Sample& s) -> bool {
            if (!opt.uniquePairs) { samples.push_back(s); return true; }
            auto k = key_of(s);
            if (seen.insert(k).second) { samples.push_back(s); return true; }
            return false;
            };

        auto gen_add_one = [&]() {
            int a = uni(rng), b = uni(rng);
            if (opt.bothOrders) {
                // choisir aléatoirement l’ordre si on ne veut qu’un seul échantillon
                if (rng() & 1ULL) std::swap(a, b);
                return try_insert({ a, b, Op::Add });
            }
            else {
                if (a > b) std::swap(a, b);
                return try_insert({ a, b, Op::Add });
            }
            };
        auto gen_sub_one = [&]() {
            int a = uni(rng), b = uni(rng);
            if (opt.nonneg) {
                if (a < b) std::swap(a, b);
                return try_insert({ a, b, Op::Sub });
            }
            else {
                if (opt.bothOrders) {
                    // choisir aléatoirement l’ordre si on ne veut qu’un seul échantillon
                    if (rng() & 1ULL) std::swap(a, b);
                    return try_insert({ a, b, Op::Sub });
                }
                else {
                    if (a < b) std::swap(a, b);
                    return try_insert({ a, b, Op::Sub });
                }
            }
            };

        if (wantAdd && wantSub && opt.balance) {
            // équilibre add/sub
            int targetAdd = opt.count / 2;
            int targetSub = opt.count - targetAdd;
            int gotAdd = 0, gotSub = 0, guard = 0;
            while ((gotAdd < targetAdd || gotSub < targetSub) && guard < opt.count * 50) {
                ++guard;
                if (gotAdd < targetAdd) gotAdd += gen_add_one() ? 1 : 0;
                if (gotSub < targetSub) gotSub += gen_sub_one() ? 1 : 0;
            }
        }
        else {
            // mélange libre (ou une seule op)
            int got = 0, guard = 0;
            while (got < opt.count && guard < opt.count * 100) {
                ++guard;
                bool doAdd;
                if (wantAdd && wantSub) doAdd = (rng() & 1ULL);
                else doAdd = wantAdd;
                bool ok = doAdd ? gen_add_one() : gen_sub_one();
                if (ok) ++got;
            }
        }

        if (opt.shuffleOut) std::shuffle(samples.begin(), samples.end(), rng);
    }

    // Écriture
    std::ofstream out(opt.outPath, std::ios::binary);
    if (!out) {
        std::cerr << "Impossible d’ouvrir " << opt.outPath << " en écriture.\n";
        return 1;
    }
    for (const auto& s : samples) {
        if (s.op == Op::Add) {
            int c = s.a + s.b;
            out << s.a << " + " << s.b << " = " << c << '\n';
        }
        else {
            int c = s.a - s.b;
            out << s.a << " - " << s.b << " = " << c << '\n';
        }
    }
    out.flush();
    if (!out) {
        std::cerr << "Erreur d’écriture sur " << opt.outPath << "\n";
        return 1;
    }

    std::cout << "OK: " << samples.size() << " lignes -> " << opt.outPath
        << "  (seed=" << seed
        << ", mode=" << opt.mode
        << ", ops=" << opt.ops
        << ", nonneg=" << (opt.nonneg ? 1 : 0)
        << ")\n";
    return 0;
}
