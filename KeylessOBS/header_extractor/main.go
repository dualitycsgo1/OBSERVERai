package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
)

type DemoInfo struct {
	MapName      string `json:"map_name"`
	ServerName   string `json:"server_name"`
	ClientName   string `json:"client_name"`
	DemoFile     string `json:"demo_file"`
	Success      bool   `json:"success"`
	ErrorMessage string `json:"error_message,omitempty"`
	DemoType     string `json:"demo_type"` // "CS:GO" or "CS2"
}

// Simple VarInt decoder for CS2 protobuf parsing
func readVarInt32(reader io.Reader) (uint32, error) {
	var result uint32
	var shift uint

	for {
		var b [1]byte
		_, err := reader.Read(b[:])
		if err != nil {
			return 0, err
		}

		result |= uint32(b[0]&0x7F) << shift
		if (b[0] & 0x80) == 0 {
			break
		}
		shift += 7
		if shift >= 32 {
			return 0, fmt.Errorf("varint too large")
		}
	}
	return result, nil
}

func removeNullBytes(s string) string {
	return strings.ReplaceAll(s, "\x00", "")
}

func removeUnicodeReplacementChars(s string) string {
	return strings.ReplaceAll(s, "\uFFFD", "")
}

// Simple protobuf string field parser for CS2 demos
func parseProtobufString(data []byte, fieldNumber int) string {
	pos := 0
	for pos < len(data) {
		if pos >= len(data) {
			break
		}

		// Read field header (tag)
		tag := data[pos]
		pos++

		wireType := tag & 0x07
		field := tag >> 3

		if wireType == 2 && int(field) == fieldNumber { // Length-delimited field
			if pos >= len(data) {
				break
			}

			// Read length
			length := int(data[pos])
			pos++

			if pos+length > len(data) {
				break
			}

			// Extract string
			return string(data[pos : pos+length])
		} else {
			// Skip this field
			switch wireType {
			case 0: // Varint
				for pos < len(data) && (data[pos]&0x80) != 0 {
					pos++
				}
				if pos < len(data) {
					pos++
				}
			case 2: // Length-delimited
				if pos >= len(data) {
					break
				}
				length := int(data[pos])
				pos++
				pos += length
			case 5: // 32-bit
				pos += 4
			default:
				// Unknown wire type, break
				break
			}
		}
	}
	return ""
}

func readCS2DemoHeader(buffer []byte) (*DemoInfo, error) {
	if len(buffer) < 128 {
		return nil, fmt.Errorf("buffer too small for CS2 header")
	}

	pos := 16 // Skip filestamp + 8 bytes

	// Read the first message type (should be 1 for DEM_FileHeader)
	if pos >= len(buffer) {
		return nil, fmt.Errorf("buffer too small to read message type")
	}

	messageType := buffer[pos]
	pos++

	if messageType != 1 {
		return nil, fmt.Errorf("unexpected first message type: %d", messageType)
	}

	// Skip tick (varint)
	for pos < len(buffer) && (buffer[pos]&0x80) != 0 {
		pos++
	}
	if pos < len(buffer) {
		pos++
	}

	// Read message size
	if pos >= len(buffer) {
		return nil, fmt.Errorf("buffer too small to read message size")
	}

	messageSize := int(buffer[pos])
	pos++

	if pos+messageSize > len(buffer) {
		return nil, fmt.Errorf("message size exceeds buffer")
	}

	// Extract the protobuf message bytes
	messageData := buffer[pos : pos+messageSize]

	// Parse protobuf fields manually (simplified)
	// Field numbers based on CS2 CDemoFileHeader:
	// 1 = demoProtocol, 2 = networkProtocol, 3 = serverName, 4 = clientName, 5 = mapName
	serverName := parseProtobufString(messageData, 3)
	clientName := parseProtobufString(messageData, 4)
	mapName := parseProtobufString(messageData, 5)

	info := &DemoInfo{
		DemoType:   "CS2",
		ServerName: removeUnicodeReplacementChars(serverName),
		ClientName: removeUnicodeReplacementChars(clientName),
		MapName:    removeUnicodeReplacementChars(mapName),
		Success:    true,
	}

	return info, nil
}

func readDemoHeader(filePath string) (*DemoInfo, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// Read first 4096 bytes to get header info
	buffer := make([]byte, 4096)
	n, err := file.Read(buffer)
	if err != nil && err != io.EOF {
		return nil, fmt.Errorf("failed to read file: %v", err)
	}
	if n < 8 {
		return nil, fmt.Errorf("file too small to be a valid demo")
	}

	// Check filestamp (first 8 bytes)
	filestamp := removeNullBytes(string(buffer[0:8]))

	info := &DemoInfo{
		DemoFile: filePath,
		Success:  false,
	}

	if filestamp == "HL2DEMO" {
		// Source 1 (CS:GO) demo format
		info.DemoType = "CS:GO"

		if n < 796 {
			return nil, fmt.Errorf("CS:GO demo header incomplete")
		}

		// Extract header fields based on CS Demo Manager offsets
		info.ServerName = removeNullBytes(string(buffer[16:276]))  // 260 bytes
		info.ClientName = removeNullBytes(string(buffer[276:536])) // 260 bytes
		info.MapName = removeNullBytes(string(buffer[536:796]))    // 260 bytes

		info.Success = true
		return info, nil

	} else if filestamp == "PBDEMS2" {
		// Source 2 (CS2) demo format - parse protobuf
		return readCS2DemoHeader(buffer)

	} else {
		return nil, fmt.Errorf("invalid demo format, filestamp: %s", filestamp)
	}
}

func main() {
	if len(os.Args) < 2 {
		log.Fatal("Usage: cs2_header_extractor.exe <demo_file_path>")
	}

	demoPath := os.Args[1]

	demoInfo, err := readDemoHeader(demoPath)
	if err != nil {
		demoInfo = &DemoInfo{
			DemoFile:     demoPath,
			Success:      false,
			ErrorMessage: err.Error(),
		}
	}

	// Output JSON result
	output, err := json.MarshalIndent(demoInfo, "", "  ")
	if err != nil {
		log.Printf("Error marshaling JSON: %v", err)
		fmt.Println(`{"success": false, "error_message": "Failed to create JSON output"}`)
	} else {
		fmt.Println(string(output))
	}
}
