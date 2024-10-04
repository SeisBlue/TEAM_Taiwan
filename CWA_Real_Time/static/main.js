const socket = io();
const traces = new Map();
const picks = new Map();
const sampleRate = 100;
const dataWindow = 15 * sampleRate;

function createTrace(traceid) {
    traces.set(traceid, Array(dataWindow).fill(0));
}

function updateTrace(traceid, newData) {
    traces.set(traceid, [newData]);
}

function createChart(traceid) {
    let chartDiv = document.createElement('div');
    chartDiv.className = 'chart';
    chartDiv.id = `chart-${traceid}`;
    document.getElementById('charts').appendChild(chartDiv);

    let currentData = traces.get(traceid);

    let data = [{
        y: [currentData],
        type: 'scatter',
        mode: 'lines'
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
}


function updateChart(traceid, currentData) {
    let update = {
        y: [currentData]
    };
    Plotly.update(`chart-${traceid}`, update);
}

function updateTime(timeStamp) {
    const timeDiv = document.getElementById('time');
    timeDiv.textContent = timeStamp;
}

function updatePick(picks) {
    const picksDiv = document.getElementById('picks');
    picksDiv.textContent = 'Picks count: ' + picks.size;
}

socket.on('connect_init', function () {
    picks.forEach((value, key) => {
        createChart(key);
    });
});

socket.on('wave_data', function (msg) {
    if (!traces.has(msg.traceid)) {
        createTrace(msg.traceid);
    }
    updateTrace(msg.traceid, msg.data);
    console.log(msg.traceid + " time: " + new Date(msg.endt).toISOString());

    updateChart(msg.traceid);

    updateTime(new Date(msg.endt).toISOString());
});

socket.on('trace_data', function (msg) {
    if (!document.getElementById(`chart-${msg.traceid}`)) {
        createChart(msg.traceid);
    }
    updateChart(msg.traceid, msg.data);
    console.log(msg.traceid + " time: " + msg.time);
});

socket.on('pick_data', function (msg) {
    if (msg.update_sec == 2) {
        if (!picks.has(msg.traceid)) {
            picks.set(msg.traceid, msg.pick_time);
            createChart(msg.traceid);
        }
    }

    if (msg.update_sec == 9) {
        if (picks.has(msg.traceid)) {
            picks.delete(msg.traceid);
            let chartDiv = document.getElementById(`chart-${msg.traceid}`);
            chartDiv.remove();
        }
    }
    updatePick(picks);
    console.log(picks);
});

