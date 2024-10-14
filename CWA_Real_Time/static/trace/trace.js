const socket = io();
const traces = new Map();
const picks = new Map();
const traceLength = 1500;


function createChart(traceid) {
    let chartDiv = document.createElement('div');
    chartDiv.className = 'chart';
    chartDiv.id = `chart-${traceid}`;
    document.getElementById('charts').appendChild(chartDiv);


    let data = [{
        y: Array(3000).fill(0),
        type: 'scatter',
        mode: 'lines',
    }];

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

        height: 50,
        margin: {t: 5, b: 20, l: 200, r: 10}
    };

    Plotly.newPlot(`chart-${traceid}`, data, layout, {displayModeBar: false});
    traces.set(traceid, Array(traceLength).fill(0));
}


function updateChart(traceid, newData) {
    let currentData = traces.get(traceid);
    currentData = [...currentData.slice(newData.length), ...newData];
    traces.set(traceid, currentData);


    let update = {
        y: [currentData]
    };

    Plotly.update(`chart-${traceid}`, update);
}

socket.on('connect_init', function () {
    traces.forEach((value, key) => {
        createChart(key);
    });
});

socket.on('wave_packet', function (msg) {

    if (!traces.has(msg.waveid)) {
        createChart(msg.waveid);
    }

    let data = msg.data
    updateChart(msg.waveid, data);

});