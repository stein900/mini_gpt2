// addgen.cpp — Génère un corpus d’additions "a + b = c" propre pour entraînement.
// - Format strict: "A + B = C\n" (1 espace autour de + et =, pas d’alignement).
// - Deux modes: grid (toutes paires) ou random (aléatoire).
// - Options: plage, nombre d’exemples, ordre (A,B) et (B,A), unicité, seed, shuffle, fichier de sortie.
// Compile: g++ -O3 -std=c++17 addgen.cpp -o addgen
// Windows (MSVC): cl /O2 /std:c++17 addgen.cpp

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

struct Args {
    int minVal = 1;
    int maxVal = 9;
    std::string mode = "random";      // "grid" | "random"
    int count = 10000;              // utilisé si mode=random
    bool bothOrders = true;         // inclure (a,b) ET (b,a)
    bool uniquePairs = false;       // random: éviter les doublons (si possible)
    bool shuffleOut = true;         // mélanger la sortie
    uint64_t seed = 0;              // 0 => auto seed
    std::string outPath = "additions.txt";
};

static void usage(const char* prog) {
    std::cerr <<
        "Usage:\n"
        "  " << prog << " [--min 1] [--max 9] [--mode grid|random]\n"
        "               [--count 10000] [--both-orders 0|1] [--unique 0|1]\n"
        "               [--shuffle 0|1] [--seed N] [--out additions.txt]\n\n"
        "Exemples:\n"
        "  # Toutes les paires 1..9 (ordre A,B et B,A), mélangées\n"
        "  " << prog << " --min 1 --max 9 --mode grid --both-orders 1 --shuffle 1 --out additions_1_9.txt\n\n"
        "  # 10k paires aléatoires 0..50, sans doublon si possible\n"
        "  " << prog << " --min 0 --max 50 --mode random --count 10000 --unique 1 --out additions_0_50_rand.txt\n";
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
    for (int i = 1;i < argc;i++) {
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
        else if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else { std::cerr << "Unknown arg: " << k << "\n"; usage(argv[0]); std::exit(1); }
    }
    if (a.minVal > a.maxVal) { std::swap(a.minVal, a.maxVal); }
    if (a.mode != "grid" && a.mode != "random") {
        std::cerr << "--mode must be 'grid' or 'random'\n"; std::exit(1);
    }
    if (a.mode == "random" && a.count <= 0) {
        std::cerr << "--count must be > 0 for random mode\n"; std::exit(1);
    }
    return a;
}

int main(int argc, char** argv) {
    auto opt = parseArgs(argc, argv);

    // RNG
    uint64_t seed = opt.seed ? opt.seed
        : (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> uni(opt.minVal, opt.maxVal);

    // Génère les paires (A,B)
    std::vector<std::pair<int, int>> pairs;

    if (opt.mode == "grid") {
        for (int a = opt.minVal; a <= opt.maxVal; ++a) {
            for (int b = opt.minVal; b <= opt.maxVal; ++b) {
                if (!opt.bothOrders && b < a) continue; // garde a<=b si on ne veut pas les deux ordres
                pairs.emplace_back(a, b);
            }
        }
        if (opt.shuffleOut) std::shuffle(pairs.begin(), pairs.end(), rng);
    }
    else {
        // random
        const int domainSize = (opt.maxVal - opt.minVal + 1) * (opt.maxVal - opt.minVal + 1);
        if (opt.uniquePairs && opt.count > domainSize) {
            std::cerr << "[warn] --unique demandé mais --count > nombre de paires possibles (" << domainSize << "). "
                "On passera en tirage avec remplacement.\n";
            opt.uniquePairs = false;
        }
        if (opt.uniquePairs) {
            // échantillonnage sans remplacement
            // on génère tout l’espace puis on prend les 'count' premiers après shuffle
            std::vector<std::pair<int, int>> all;
            all.reserve(domainSize);
            for (int a = opt.minVal; a <= opt.maxVal; ++a)
                for (int b = opt.minVal; b <= opt.maxVal; ++b)
                    all.emplace_back(a, b);
            std::shuffle(all.begin(), all.end(), rng);
            if ((int)all.size() > opt.count) all.resize(opt.count);
            pairs = std::move(all);
        }
        else {
            // avec remplacement
            pairs.reserve(opt.count);
            for (int i = 0;i < opt.count;i++) {
                int a = uni(rng);
                int b = uni(rng);
                pairs.emplace_back(a, b);
            }
        }
    }

    // Écrit le fichier — format STRICT: "A + B = C\n"
    std::ofstream out(opt.outPath, std::ios::binary);
    if (!out) {
        std::cerr << "Impossible d’ouvrir " << opt.outPath << " en écriture.\n";
        return 1;
    }
    for (const auto& p : pairs) {
        int a = p.first, b = p.second, c = a + b;
        // aucun alignement, aucun espace superflu :
        // [a][space][+][space][b][space][=][space][c][\n]
        out << a << " + " << b << " = " << c << '\n';
    }
    out.flush();
    if (!out) {
        std::cerr << "Erreur d’écriture sur " << opt.outPath << "\n";
        return 1;
    }

    std::cout << "OK: " << pairs.size() << " lignes -> " << opt.outPath
        << "  (seed=" << seed << ", mode=" << opt.mode << ")\n";
    return 0;
}
