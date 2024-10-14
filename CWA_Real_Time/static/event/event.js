const socket = io();
const traces = new Map();
const times = new Map();
const picks = new Map();


function createChart(traceid) {
    let chartDiv = document.createElement('div');
    chartDiv.className = 'chart';
    chartDiv.id = `chart-${traceid}`;
    document.getElementById('charts').appendChild(chartDiv);

    let ch_z = {
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines',
        name: 'Z'
    };
    let ch_n = {
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines',
        name: 'N'
    };
    let ch_e = {
        x: [],
        y: [],
        type: 'scatter',
        mode: 'lines',
        name: 'E'
    };
    let data = [ch_z, ch_n, ch_e];

    let layout = {
        title: {
            text: `${traceid}`,
            xanchor: 'left',
            yanchor: 'middle',
            x: 0,   // x = 0 表示最左邊
            y: 0.5, // y = 0.5 表示垂直居中
            standoff: 20,  // 離圖表邊界的距離
            font: {
                size: 12  // 設定標題字體大小為 12px
            }
        },
        xaxis: {
            type: 'date',  // 設定 x 軸為時間
            tickformat: '%H:%M:%S',  // 設定時間格式
        },
        height: 50,
        margin: {t: 5, b: 20, l: 200, r: 10}
    };

    Plotly.newPlot(`chart-${traceid}`, data, layout, {displayModeBar: false});
}


function updateChart(traceid, currentTime, currentData) {
let update = {
    x: [currentTime, currentTime, currentTime],
    y: [currentData.z, currentData.n, currentData.e]
};
    console.log(update);

    Plotly.update(`chart-${traceid}`, update);
}

socket.on('connect_init', function () {
    picks.forEach((value, key) => {
        createChart(key);
    });
});

socket.on('event_data', function (msg) {
    // 清空所有圖表
    let chartsdiv = document.getElementById('charts');

    while (chartsdiv.firstChild) {
        chartsdiv.removeChild(chartsdiv.lastChild);
    }

    // 畫出所有資料
    for (const [key, value] of Object.entries(msg)) {
        let traceid = key;
        let trace = value.trace;
        let data = trace.data
        let time = trace.time.map(time => new Date(time * 1000).toISOString());


        createChart(traceid);
        updateChart(traceid, time, data);
    }
});
