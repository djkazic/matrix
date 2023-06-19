package main

import (
	"bytes"
	"encoding/csv"
	"encoding/hex"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/btcsuite/btcd/chaincfg/chainhash"
	"github.com/btcsuite/btcd/rpcclient"
	"github.com/btcsuite/btcutil"
	"github.com/btcsuite/btcd/wire"
	"github.com/joho/godotenv"
	zmq "github.com/pebbe/zmq4"
)

type TxData struct {
	Height  int32
	Time    time.Time
	FeeRate float64
}

var (
	txHeightMap = make(map[string]TxData)
	mutex       sync.RWMutex
)

func main() {
	fmt.Println("starting matrix data logger")
	err := godotenv.Load(".env")
	if err != nil {
		log.Fatal("Error loading .env file")
	}
    file, err := os.OpenFile("training_data.csv", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("Error opening CSV file: %v", err)
	}
	defer file.Close()
	go func() {
		if err := listen(os.Getenv("ZMQ_TX"), "rawtx"); err != nil {
			log.Fatal(err)
		}
	}()
	go func() {
		if err := listen(os.Getenv("ZMQ_BLOCK"), "rawblock"); err != nil {
			log.Fatal(err)
		}
	}()
	select {}
}

func listen(address, subTopic string) error {
	socket, err := zmq.NewSocket(zmq.SUB)
	if err != nil {
		return err
	}
	defer socket.Close()
	err = socket.Connect(address)
	if err != nil {
		return err
	}
	err = socket.SetSubscribe(subTopic)
	if err != nil {
		return err
	}
	rpcClient, err := newRPCClient(os.Getenv("RPC_URI"), os.Getenv("RPC_USER"), os.Getenv("RPC_PASSWORD"))
	if err != nil {
		return err
	}
	defer rpcClient.Shutdown()
	for {
		// receive all parts of the message
		msgParts, err := socket.RecvMessageBytes(0)
		if err != nil {
			log.Fatalf("recv error: %v", err)
		}

		topic := string(msgParts[0])
		var msg []byte
		switch topic {
		case "rawtx":
			// concatenate all parts of the message except the topic into a single byte slice
			for _, part := range msgParts[1:] {
				msg = append(msg, part...)
			}
			tx, err := btcutil.NewTxFromBytes(msg)
			if err != nil && err != io.EOF {
				log.Fatalf("rawtx parse error: %v", err)
			}
			if tx == nil {
				continue
			}
			msgTx := tx.MsgTx()
			// now := time.Now()
			if msgTx != nil {
				txHex := hex.EncodeToString(msg)
				_, feeRate, err := getTransactionDetails(rpcClient, txHex)
				if feeRate < 0 {
					//log.Printf("Skipping transaction %s invalid feeRate", msgTx.TxHash())
                    continue
				}
				if err != nil {
					log.Fatalf("Error getting transaction %s vsize: %v", msgTx.TxHash(), err)
				}
				mutex.Lock()
				txHeightMap[msgTx.TxHash().String()] = TxData{
					Height: getBestChainHeight(rpcClient),
					Time: time.Now(),
					FeeRate: feeRate,
				}
				mutex.Unlock()
			}

		case "rawblock":
			// concatenate all parts of the message except the topic into a single byte slice
			for _, part := range msgParts[1:] {
				msg = append(msg, part...)
			}

			log.Printf("received rawblock data length: %d", len(msg))
			block, err := btcutil.NewBlockFromBytes(msg)
			if err != nil {
				log.Printf("rawblock parse error: %v", err)
				continue
			}
			if block == nil {
				log.Printf("rawblock block nil: %x", msg)
				continue
			}
			processBlock(rpcClient, block)
		}
	}
}

func processBlock(client *rpcclient.Client, block *btcutil.Block) {
	blockHash := block.Hash()
	blockInfo, err := client.GetBlockVerbose(blockHash)
	if err != nil {
		log.Fatalf("getblock error: %v", err)
	}
	blockHeight := int32(blockInfo.Height)
	cpfpTxs := make(map[string]*wire.MsgTx)
	// Iterate through the transactions in the block
	for _, tx := range block.Transactions() {
		txHash := tx.Hash().String()
		isCPFP, err := isCPFPTransaction(client, tx)
		if err != nil {
			log.Fatal(err)
		}
		if isCPFP {
			cpfpTxs[txHash] = tx.MsgTx()
		}
		mutex.Lock()
		txData, exists := txHeightMap[txHash]
                // todo: delete map entrt
		mutex.Unlock()
		if exists {
			blockHeightDiff := blockHeight - txData.Height
			if blockHeightDiff < 0 {
				blockHeightDiff = 0
			}
			if isCPFP {
				fmt.Printf("Transaction %s confirmed at block height %d feerate %f (blockHeight Difference: %d)\n", txHash, blockHeight, txData.FeeRate, blockHeightDiff)
			}
			packageFeeRate := calculatePackageFeeRate(client, tx, cpfpTxs)
			if blockHeightDiff > 0 {
				err = writeToCSV(txHash, txData.Time, blockHeightDiff, packageFeeRate)
				if err != nil {
					log.Fatal(err)
				}
			}
		}
	}
}

func newRPCClient(rpcURL, rpcUser, rpcPassword string) (*rpcclient.Client, error) {
	connCfg := &rpcclient.ConnConfig{
		Host:         rpcURL,
		User:         rpcUser,
		Pass:         rpcPassword,
		HTTPPostMode: true,
		DisableTLS:   true,
	}
	rpcClient, err := rpcclient.New(connCfg, nil)
	if err != nil {
		return nil, err
	}
	return rpcClient, nil
}

func findPreviousOutput(client *rpcclient.Client, prevTxHash *chainhash.Hash, prevOutputIndex uint32) (*wire.TxOut, error) {
	rawTx, err := client.GetRawTransactionVerbose(prevTxHash)
	if err != nil {
		return nil, err
	}
	if prevOutputIndex >= uint32(len(rawTx.Vout)) {
		return nil, fmt.Errorf("previous output index out of range")
	}
	txOut := rawTx.Vout[prevOutputIndex]
	script, err := hex.DecodeString(txOut.ScriptPubKey.Hex)
	if err != nil {
		return nil, err
	}

	return &wire.TxOut{
		Value:    int64(txOut.Value * 1e8), // Convert from BTC to satoshis
		PkScript: script,
	}, nil
}

func getTransactionDetails(client *rpcclient.Client, txHex string) (float64, float64, error) {
	txBytes, err := hex.DecodeString(txHex)
	if err != nil {
		return 0, 0, err
	}
	msgTx := wire.NewMsgTx(wire.TxVersion)
	err = msgTx.Deserialize(bytes.NewReader(txBytes))
	if err != nil {
		return 0, 0, err
	}
	vsize := calculateVSize(msgTx)
	totalFee := calculateTotalFee(client, msgTx)
	feeRate := totalFee / vsize
	return vsize, feeRate, nil
}

func calculateTotalFee(client *rpcclient.Client, tx *wire.MsgTx) float64 {
	var totalInputValue float64
	for _, txIn := range tx.TxIn {
		prevTxOut := txIn.PreviousOutPoint
		// Skip coinbase transaction inputs
		if prevTxOut == *wire.NewOutPoint(&chainhash.Hash{}, wire.MaxPrevOutIndex) {
			continue
		}
		// Retrieve the previous transaction's output
		txOut, err := findPreviousOutput(client, &prevTxOut.Hash, prevTxOut.Index)
		if err != nil {
			if strings.Contains(err.Error(), "No such mempool or blockchain transaction") {
				continue
			}
			return 0
		}
		if txOut == nil {
			return 0
		}
		totalInputValue += float64(txOut.Value)
	}
	var totalOutputValue float64
	for _, txOut := range tx.TxOut {
		totalOutputValue += float64(txOut.Value)
	}
	totalFee := totalInputValue - totalOutputValue
	return totalFee
}

func calculateVSize(tx *wire.MsgTx) float64 {
	baseSize := tx.SerializeSizeStripped()
	totalSize := tx.SerializeSize()
	witnessSize := totalSize - baseSize
	vsize := float64(baseSize) + float64(witnessSize)/4
	return vsize
}

func writeToCSV(txHash string, date time.Time, blockDiff int32, feeRate float64) error {
	file, err := os.OpenFile("training_data.csv", os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		return err
	}
	defer file.Close()
	_, err = file.Seek(0, io.SeekEnd) // Seek to the end of the file
	if err != nil {
		return err
	}
	writer := csv.NewWriter(file)
	defer writer.Flush()
	row := []string{
		date.Format("2006-01-02 15:04:05"),      // Date formatted as "YYYY-MM-DD HH:MM:SS"
		strconv.FormatInt(int64(blockDiff), 10), // Block diff
		fmt.Sprintf("%.2f", feeRate),            // Fee rate formatted with 2 decimal places
	}
	err = writer.Write(row)
	if err != nil {
		return err
	}
	return nil
}

func isCPFPTransaction(client *rpcclient.Client, tx *btcutil.Tx) (bool, error) {
	// Check if the transaction has a parent transaction in the mempool
	for _, txIn := range tx.MsgTx().TxIn {
		prevTxHash := &txIn.PreviousOutPoint.Hash
		if prevTxHash != nil && !prevTxHash.IsEqual(&chainhash.Hash{}) {
			_, err := client.GetRawTransactionVerbose(prevTxHash)
			if err != nil && strings.Contains(err.Error(), "No such mempool or blockchain transaction") {
				return true, nil
			} else if err != nil {
				return false, fmt.Errorf("error retrieving previous transaction %s: %v", prevTxHash, err)
			}
		}
	}
	return false, nil
}

func isTransactionInMempool(client *rpcclient.Client, txHash chainhash.Hash) (bool, error) {
	tx, err := client.GetRawTransaction(&txHash)
	if err != nil {
		// If the error message indicates that the transaction is not found in the mempool,
		// we can consider it as not being in the mempool
		if strings.Contains(err.Error(), "No such mempool transaction") {
			return false, nil
		}
		return false, err
	}
	return tx != nil, nil
}

func calculatePackageFeeRate(client *rpcclient.Client, tx *btcutil.Tx, cpfpTxs map[string]*wire.MsgTx) float64 {
	totalFee := 0.0
	totalVSize := 0.0
	// Calculate the total fee and total vsize of the package
	calculateFee := func(msgTx *wire.MsgTx) {
		vsize := calculateVSize(msgTx)
		totalVSize += vsize

		fee := calculateTotalFee(client, msgTx)
		totalFee += fee
	}
	// Add the current transaction's fee to the total
	calculateFee(tx.MsgTx())
	// Add the fees of all CPFP transactions in the package
	for _, cpfpTx := range cpfpTxs {
		calculateFee(cpfpTx)
	}
	// Calculate the package feerate
	packageFeeRate := totalFee / totalVSize
	return packageFeeRate
}

func getBestChainHeight(client *rpcclient.Client) int32 {
	info, err := client.GetBlockChainInfo()
	if err != nil {
		log.Fatalf("Error getting blockchain info: %v", err)
	}

	return info.Blocks
}
