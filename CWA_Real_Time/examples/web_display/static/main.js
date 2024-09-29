const socket = io();
const stations = new Map();
const picks = new Map();
const sampleRate = 100;
const dataWindow = 15 * sampleRate;

function createTrace(station) {
    stations.set(station, Array(dataWindow).fill(0));
}

function updateTrace(station, newData) {
    let currentData = stations.get(station);
    currentData.splice(0, newData.length); // Remove the oldest data
    currentData.push(...newData); // Add the new data
    stations.set(station, currentData);
}

function createChart(station) {
    let chartDiv = document.createElement('div');
    chartDiv.className = 'chart';
    chartDiv.id = `chart-${station}`;
    document.getElementById('charts').appendChild(chartDiv);

    let currentData = stations.get(station);

    let data = [{
        y: [currentData],
        type: 'scatter',
        mode: 'lines'
    }];

    let layout = {
        title: {
            text: `${station}`,
            xanchor: 'left',
            yanchor: 'middle',
            x: 0,   // x = 0 表示最左邊
            y: 0.5, // y = 0.5 表示垂直居中
            standoff: 20,  // 離圖表邊界的距離
            font: {
                size: 12  // 設定標題字體大小為 12px
            }
        },

        height: 50,
        margin: {t: 5, b: 20, l: 200, r: 10}
    };

    Plotly.newPlot(`chart-${station}`, data, layout, {displayModeBar: false});
}


function updateChart(station) {
    let currentData = stations.get(station);
    let update = {
        y: [currentData]
    };
    Plotly.update(`chart-${station}`, update);
}

socket.on('connect_init', function () {
    picks.forEach((value, key) => {
        createChart(key);
    });
});

socket.on('earthquake_data', function (msg) {
    if (!stations.has(msg.station)) {
        createTrace(msg.station);
    }
    updateTrace(msg.station, msg.data);
    console.log(msg.station);

    if (picks.has(msg.station)) {
        updateChart(msg.station);
    }

});

socket.on('pick_data', function (msg) {
    if (msg.repeat == 2) {
        if (!picks.has(msg.station)) {
            picks.set(msg.station, msg.pick_time);
            createChart(msg.station);
        }
    }

    if (msg.repeat == 9) {
        if (picks.has(msg.station)) {
            picks.delete(msg.station);
            let chartDiv = document.getElementById(`chart-${msg.station}`);
            chartDiv.remove();
        }
    }


    console.log(picks);
});

