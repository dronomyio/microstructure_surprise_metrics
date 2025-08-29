#include "polygon_parser.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <cstring>
#include <stdlib.h>

#ifdef HAS_ZLIB
#include <zlib.h>
#endif

#ifdef HAS_CURL
#include <curl/curl.h>
#endif

namespace surprise_metrics {

class PolygonDataLoader {
private:
    std::string api_key_;
    std::string base_url_ = "https://api.polygon.io/v3/files/flatfiles";
    
    struct MemoryStruct {
        char* memory;
        size_t size;
    };
    
#ifdef HAS_CURL
    static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
        size_t realsize = size * nmemb;
        struct MemoryStruct* mem = (struct MemoryStruct*)userp;
        
        char* ptr = (char*)realloc(mem->memory, mem->size + realsize + 1);
        if (!ptr) {
            return 0;
        }
        
        mem->memory = ptr;
        memcpy(&(mem->memory[mem->size]), contents, realsize);
        mem->size += realsize;
        mem->memory[mem->size] = 0;
        
        return realsize;
    }
#endif
    
public:
    PolygonDataLoader(const std::string& api_key) : api_key_(api_key) {}
    
    // Download flat files for a specific date
    std::string download_trades_file(const std::string& date, const std::string& ticker) {
        std::string url = base_url_ + "/us/stocks/trades/" + date + "/" + ticker + ".csv.gz";
        return download_file(url);
    }
    
    std::string download_quotes_file(const std::string& date, const std::string& ticker) {
        std::string url = base_url_ + "/us/stocks/quotes/" + date + "/" + ticker + ".csv.gz";
        return download_file(url);
    }
    
private:
    std::string download_file(const std::string& url) {
#ifdef HAS_CURL
        CURL* curl;
        CURLcode res;
        struct MemoryStruct chunk;
        
        chunk.memory = (char*)malloc(1);
        chunk.size = 0;
        
        curl = curl_easy_init();
        if (curl) {
            std::string auth_url = url + "?apiKey=" + api_key_;
            
            curl_easy_setopt(curl, CURLOPT_URL, auth_url.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&chunk);
            curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
            
            res = curl_easy_perform(curl);
            
            if (res != CURLE_OK) {
                fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
            }
            
            curl_easy_cleanup(curl);
        }
        
        std::string result(chunk.memory, chunk.size);
        free(chunk.memory);
        
        return result;
#else
        std::cerr << "CURL not available. Please install libcurl-dev to download data.\n";
        std::cerr << "You can manually download files from: " << url << "\n";
        return "";
#endif
    }
};

// Decompress gzip data
std::string decompress_gzip(const std::string& compressed_data) {
#ifdef HAS_ZLIB
    z_stream stream;
    memset(&stream, 0, sizeof(stream));
    stream.zalloc = Z_NULL;
    stream.zfree = Z_NULL;
    stream.opaque = Z_NULL;
    stream.avail_in = compressed_data.size();
    stream.next_in = (Bytef*)compressed_data.data();
    
    if (inflateInit2(&stream, 16 + MAX_WBITS) != Z_OK) {
        return "";
    }
    
    std::string decompressed;
    char buffer[4096];
    
    do {
        stream.avail_out = sizeof(buffer);
        stream.next_out = reinterpret_cast<Bytef*>(buffer);
        
        int ret = inflate(&stream, Z_NO_FLUSH);
        if (ret == Z_STREAM_ERROR) {
            inflateEnd(&stream);
            return "";
        }
        
        decompressed.append(buffer, sizeof(buffer) - stream.avail_out);
    } while (stream.avail_out == 0);
    
    inflateEnd(&stream);
    return decompressed;
#else
    std::cerr << "ZLIB not available. Cannot decompress gzip files.\n";
    return compressed_data; // Return as-is, hoping it's not actually compressed
#endif
}

// Parse Polygon trades data format
std::vector<Trade> parse_polygon_trades(const std::string& csv_data) {
    std::vector<Trade> trades;
    
    // Parse CSV with nanosecond timestamps
    // Format: participant_timestamp,sip_timestamp,trf_timestamp,sequence,symbol,size,price,conditions,tape
    
    std::istringstream stream(csv_data);
    std::string line;
    
    // Skip header
    std::getline(stream, line);
    
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        std::string field;
        Trade trade;
        
        try {
            // Parse participant_timestamp (nanoseconds since Unix epoch)
            std::getline(iss, field, ',');
            trade.timestamp = std::chrono::nanoseconds(std::stoull(field));
            
            // Skip sip_timestamp and trf_timestamp
            std::getline(iss, field, ',');
            std::getline(iss, field, ',');
            
            // Skip sequence
            std::getline(iss, field, ',');
            
            // Skip symbol (already filtered)
            std::getline(iss, field, ',');
            
            // Parse size
            std::getline(iss, field, ',');
            trade.size = std::stoull(field);
            
            // Parse price
            std::getline(iss, field, ',');
            trade.price = std::stod(field);
            
            // Parse conditions
            std::getline(iss, field, ',');
            // Parse condition codes (up to 4)
            for (int i = 0; i < 4 && i < field.length(); ++i) {
                trade.conditions[i] = field[i];
            }
            
            // Parse tape (exchange)
            std::getline(iss, field, ',');
            if (!field.empty()) {
                trade.exchange = field[0];
            }
            
            trades.push_back(trade);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing trade: " << e.what() << std::endl;
            continue;
        }
    }
    
    return trades;
}

// Parse Polygon quotes data format
std::vector<Quote> parse_polygon_quotes(const std::string& csv_data) {
    std::vector<Quote> quotes;
    
    // Parse CSV with nanosecond timestamps
    // Format: participant_timestamp,sip_timestamp,trf_timestamp,sequence,symbol,
    //         bid_price,bid_size,bid_exchange,ask_price,ask_size,ask_exchange,tape
    
    std::istringstream stream(csv_data);
    std::string line;
    
    // Skip header
    std::getline(stream, line);
    
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        std::string field;
        Quote quote;
        
        try {
            // Parse participant_timestamp (nanoseconds)
            std::getline(iss, field, ',');
            quote.timestamp = std::chrono::nanoseconds(std::stoull(field));
            
            // Skip sip_timestamp and trf_timestamp
            std::getline(iss, field, ',');
            std::getline(iss, field, ',');
            
            // Skip sequence
            std::getline(iss, field, ',');
            
            // Skip symbol
            std::getline(iss, field, ',');
            
            // Parse bid_price
            std::getline(iss, field, ',');
            quote.bid_price = std::stod(field);
            
            // Parse bid_size
            std::getline(iss, field, ',');
            quote.bid_size = std::stoull(field);
            
            // Parse bid_exchange
            std::getline(iss, field, ',');
            if (!field.empty()) {
                quote.bid_exchange = field[0];
            }
            
            // Parse ask_price
            std::getline(iss, field, ',');
            quote.ask_price = std::stod(field);
            
            // Parse ask_size
            std::getline(iss, field, ',');
            quote.ask_size = std::stoull(field);
            
            // Parse ask_exchange
            std::getline(iss, field, ',');
            if (!field.empty()) {
                quote.ask_exchange = field[0];
            }
            
            quotes.push_back(quote);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing quote: " << e.what() << std::endl;
            continue;
        }
    }
    
    return quotes;
}

// Load trades from file
std::vector<Trade> load_trades_from_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return {};
    }
    
    // Read entire file into string
    std::string data((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
    file.close();
    
    // Check if it's gzipped (magic number: 1f 8b)
    bool is_gzipped = false;
    if (data.size() >= 2) {
        is_gzipped = (static_cast<unsigned char>(data[0]) == 0x1f && 
                      static_cast<unsigned char>(data[1]) == 0x8b);
    }
    
    if (is_gzipped) {
        data = decompress_gzip(data);
    }
    
    return parse_polygon_trades(data);
}

// Load quotes from file
std::vector<Quote> load_quotes_from_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return {};
    }
    
    // Read entire file into string
    std::string data((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
    file.close();
    
    // Check if it's gzipped
    bool is_gzipped = false;
    if (data.size() >= 2) {
        is_gzipped = (static_cast<unsigned char>(data[0]) == 0x1f && 
                      static_cast<unsigned char>(data[1]) == 0x8b);
    }
    
    if (is_gzipped) {
        data = decompress_gzip(data);
    }
    
    return parse_polygon_quotes(data);
}

} // namespace surprise_metrics
