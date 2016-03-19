var CHART_PADDING = 30;

Chart.types.Scatter.extend({
  name: "VideoChart",
  initialize: function(data){
    console.log('My Line chart extension');
    Chart.types.Scatter.prototype.initialize.apply(this, arguments);

    var padding = 10;
    this.horX = 0;
    this.yL = this.chart.ctx.canvas.height - 22*padding
    this.sentimentEvents = [];
    this.dataLength = data[0]['data'].length;
  },
  draw: function(){
    Chart.types.Scatter.prototype.draw.apply(this, arguments);
    // TODO
    var x = this.horX;
    var scale = this.scale;
    this.chart.ctx.beginPath();
    this.chart.ctx.moveTo(x, 0 );
    this.chart.ctx.strokeStyle = '#ff0000';
    //HACKY HACK
    this.chart.ctx.lineTo(x, this.yL);
    this.chart.ctx.stroke();
    this.drawBox();
  },
  updateLine: function(x){
    // Erase line and redraw it
    this.horX = x;
    this.draw();
  },
  drawBox : function(){
    var that = this;
    this.sentimentEvents.forEach(function(sEvent) {
      var startF = sEvent['startFrame'];
      var endF = sEvent['endFrame'];
      var sentiment = sEvent['sentimentGood'];
      var ctx = that.chart.ctx;
      var yU = 0;
      var canvasWidth = $('#' +  ctx.canvas.id).width();
      var xL = startF / that.dataLength * (canvasWidth - CHART_PADDING) + CHART_PADDING - 10;
      var xR = endF / that.dataLength * (canvasWidth - CHART_PADDING) + CHART_PADDING - 10;
      if (sentiment){
        ctx.fillStyle = "rgba(102, 204, 0, 0.2)";
      } else{
        ctx.fillStyle = "rgba(255, 102, 102, 0.2)";
      }
      ctx.fillRect(xL,yU,xR - xL,that.yL - yU);

    });
  },
  setSentimentEvents : function(sentimentEvents) {
    this.sentimentEvents = sentimentEvents;
  }

});

